# GGUF files of llama.cpp

## What it does

`AutoModelForCausalLM.FromGguf(path)` and `Qwen3_5ForCausalLM.FromGguf(path)` load a language model from a GGUF file of llama.cpp. No ONNX export and no `.sentis` file: the graph is built in C#, and the tensors go from the file to the GPU one at a time.

Three packages share the work:

| Package | Part |
|---|---|
| tools.doji.formats.gguf | Reads the header and the tensors (`GgufFile`). Converts tensor types to float32 or float16 with Burst jobs (`GgufConvert`). No Inference Engine dependency. |
| tools.doji.sentis-utils | `ExternalWeights` and `IWeightSource`: a graph whose weights come from any source, compiled to a `ModelFile`. `BlockQuant`: MatMul and Gather layers that read Q8_0, Q4_K, Q5_K and Q6_K blocks on the device (see `internal-docs/external-weights.md` and `block-quant.md` there). |
| tools.doji.transformers | The config and the graph of each architecture (`Qwen3_5Gguf`), the weight source over a `GgufFile` (`GgufWeights`), and `FromGguf`. |

## How a load works

1. `GgufFile.Open` reads the metadata and the tensor table. The tensor data stays in the file.
2. `Qwen3_5Gguf.Config` maps the metadata (`qwen35.*`) to a `PretrainedConfig`, and `GgufModels.GenerationConfig` takes the special tokens from `tokenizer.ggml.*`.
3. `Qwen3_5Gguf.Build` builds the graph with `FunctionalGraph`, with the inputs and outputs of the ONNX export (`input_ids`, `past_conv.{i}`, `present.{i}.key`, ...). So `PreTrainedModel`, the cache and generation work as for a `.sentis` file.
4. Each weight is an external weight. Linear and embedding weights of type Q8_0, Q4_K, Q5_K or Q6_K stay in their blocks: `GgufWeights.MatMul` and `GgufWeights.Embedding` read them with `BlockQuantMatMul` and `BlockQuantGather`, and the source repacks the ggml blocks (`BlockQuant.Pack`) as it reads them. A tied head and the embedding share one packed tensor. Other tensors are converted: F32 stays float32, all other types (F16, BF16, other quantized types) become float16 and are cast to float32 where a layer reads them, and linear weights are transposed to [in, out]. Quantized tensors that become float16 log a warning.
5. `ExternalWeights.Compile` returns a `ModelFile`. Creating the worker converts each weight into the staging buffer of the upload and uploads it, so the weights are never all in RAM. A model with float16 weights gets the float16 KV cache, as a float16 `.sentis` file does.

## Differences from transformers

- transformers loads a GGUF file with `from_pretrained(repo_id, gguf_file=...)` and dequantizes every tensor to float32 (or the given `dtype`) into a PyTorch model. Here, `FromGguf` takes a local path: AI Model Hub does not deliver GGUF files yet. Q8_0, Q4_K, Q5_K and Q6_K weights stay quantized on the device, as llama.cpp keeps them, and the kernels compute in float32 with the dequantized values. Other tensors that are not F32 become float16 on the device, not float32.
- transformers maps GGUF names and metadata to its own checkpoints with tables in `integrations/ggml.py`. It has no table for `qwen35`. `Qwen3_5Gguf` undoes what `convert_hf_to_gguf.py` changes instead:
  - The norm weights of `Qwen3_5RMSNorm` hold 1 + weight. The gated norm of the linear attention (`ssm_norm`) holds its weight.
  - `ssm_a` holds -exp(A_log). `ssm_conv1d` is squeezed to [channels, kernel].
  - With fewer key heads than value heads, the converter puts the value heads in tiled order: value head j uses key head j % key heads. transformers groups them by key head (`repeat_interleave`). The graph repeats the query and key heads tiled (concatenated) instead. The recurrent state is then in tiled order too. The cache never looks inside it, so nothing else changes.
  - Newer converters merge `attn_qkv` and `attn_gate` into `attn_qkv_gate`, and `ssm_beta` and `ssm_alpha` into `ssm_beta_alpha` (rows of the first, then of the second). The graph runs one MatMul on the merged weight and slices the result. It reads the older separate tensors too.
  - The rotary embedding is interleaved MRoPE. For text, the three position sections are equal, so it is the plain rotary embedding of the first `rope.dimension_count` values of each head.
- The tokenizer still comes from the Hugging Face files. `tokenizer.ggml.*` holds everything to build a `tokenizer.json`, except the pre-tokenizer, which llama.cpp knows by the name `tokenizer.ggml.pre`.

## Tests

`GgufCausalLMTest` loads three GGUF files of the tiny Qwen3.5 model (`--outtype f32`, `bf16` and `q8_0` of `convert_hf_to_gguf.py`, `--no-mtp`). The tiny model has 2 key heads and 4 value heads, so the tests cover the tiled order. The F32 file has the weights of the tiny model, so its reference data is that of the ONNX export. For BF16 and Q8_0, `tools/causal_lm/make_models.py gguf-references` rounds the matrices as the file holds them (and BF16 further to float16, as Unity converts it), and runs transformers. The Q8_0 file covers the packed path on the GPU and the CPU. The tiny model's rows of 32 values are too short for the K-quants; the layer tests of sentis-utils cover those. The files are in [julienkay/tiny-random-Qwen3_5ForCausalLM-GGUF](https://huggingface.co/julienkay/tiny-random-Qwen3_5ForCausalLM-GGUF). AI Model Hub does not serve GGUF files yet: the tests read them from the folder in the environment variable `DOJI_TINY_QWEN3_5_GGUF` and are ignored without it.

## Limits

- Only the architecture `qwen35`.
- Q4_0, Q4_1, Q5_0, Q5_1, Q2_K and Q3_K tensors load as float16. Other types (the IQ types, for example) are not converted.
- The prefill kernels of BlockQuant are naive: they do not tile the input rows.
- A tied head of a float16 table (no `output.weight`, embedding not in a packed type) is a second, transposed float16 copy of the table.
