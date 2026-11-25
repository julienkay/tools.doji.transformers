# GGUF files of llama.cpp

## What it does

`AutoModelForCausalLM.FromGguf(path)` and the `FromGguf` of `LlamaForCausalLM`, `Qwen3ForCausalLM`, `Phi3ForCausalLM` and `Qwen3_5ForCausalLM` load a language model from a GGUF file of llama.cpp. Architectures: `llama` (for example SmolLM2), `qwen3`, `phi3` and `qwen35`. No ONNX export and no `.sentis` file: the graph is built in C#, and the tensors go from the file to the GPU one at a time.

Three packages share the work:

| Package | Part |
|---|---|
| tools.doji.formats.gguf | Reads the header and the tensors (`GgufFile`). Converts tensor types to float32 or float16 with Burst jobs (`GgufConvert`). No Inference Engine dependency. |
| tools.doji.sentis-utils | `ExternalWeights` and `IWeightSource`: a graph whose weights come from any source, compiled to a `ModelFile`. `BlockQuant`: MatMul and Gather layers that read Q8_0, Q4_K, Q5_K and Q6_K blocks on the device (see `internal-docs/external-weights.md` and `block-quant.md` there). |
| tools.doji.transformers | The config and the graph of each architecture (`LlamaGguf` for `llama`, `qwen3` and `phi3`, `Qwen3_5Gguf` for `qwen35`), the weight source over a `GgufFile` (`GgufWeights`), the tokenizer (`GgufTokenizer`), and `FromGguf`. |

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
- `LlamaGguf` undoes what the converter changes for `llama`, `qwen3` and `phi3`:
  - For `llama`, the converter permutes the rows of the query and key weights of each head: row 2j holds row j, and row 2j + 1 holds row j + head_dim / 2. llama.cpp then rotates pairs of neighbor values. The graph does the same. The query and the key are permuted alike, so their dot products are those of transformers. The keys in the cache are in the permuted order; the cache does not look inside them.
  - For `phi3`, `attn_qkv` holds the rows of the query, key and value weights, and `ffn_up` the rows of the gate and up weights, as in transformers. The graph slices the results. A `phi3.attention.sliding_window` above 0 masks keys a window or more before the query, as transformers' sliding window mask.
  - For Llama 3.1 and later, `rope_freqs.weight` holds the factors of the llama3 rope scaling. The graph divides each frequency by its factor.
- transformers builds the tokenizer of a GGUF file with `convert_gguf_tokenizer` and a converter class per architecture. `AutoTokenizer.FromGguf` builds a `tokenizer.json` from `tokenizer.ggml.*` and loads it with `TokenizersBackend` (see below).

## Tokenizer

`GgufTokenizer` builds a `tokenizer.json` and a `TokenizerConfig` from the metadata:

- `tokenizer.ggml.model` gpt2 (byte-level BPE): `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` give the vocabulary and merges.
- `tokenizer.ggml.model` llama (sentencepiece BPE, for example Phi-3): a `LlamaTokenizer`. The file has no merges, so they come from the vocabulary, ranked by `tokenizer.ggml.scores`, as transformers' `generate_merges` with scores. `tokenizer.ggml.add_space_prefix` (true when missing) sets `add_prefix_space`. For `phi3`, the prefix space comes from a normalizer (Prepend and Replace) and there is no pre-tokenizer, as in the `tokenizer.json` of Phi-3: `"  hi"` becomes `▁▁▁hi`, not `▁▁hi`.
- `tokenizer.ggml.token_type` gives the added tokens: control tokens (3) are special added tokens, user-defined tokens (4) are added tokens that are not special. Neither is normalized. Unused tokens (5), such as the `[PAD{i}]` tokens that pad the vocabulary of Phi-3, are left out of the vocabulary.
- The metadata has no pre-tokenizer. llama.cpp knows it by its name in `tokenizer.ggml.pre` and has the regex in its code. `GgufTokenizer.PreTokenizers` maps each supported name (`qwen35`, `qwen2`, `smollm`) to the normalizer, pre-tokenizer, post-processor and decoder of the `tokenizer.json` of the models that use it. A file with another name throws `NotSupportedException`.
- The special tokens come from `tokenizer.ggml.*_token_id`, `add_bos_token` and `add_eos_token` from the keys of the same names (only when true), and the chat template from `tokenizer.chat_template`. `convert_hf_to_gguf.py` sets the end-of-sequence token of Qwen3.5 to `<|im_end|>`, so generation stops at the end of a chat turn.

transformers' own GGUF tokenizers (`integrations/gguf/gguf_tokenizer_mapping.py`) differ: for gpt2 they take a `GPT2Tokenizer` and replace its pre-tokenizer with the regex of the `tokenizer.ggml.pre` name, if transformers knows the name. So they have no NFC normalizer for `qwen35` and `qwen2`, and no single-digit split for `smollm`. `GgufTokenizer` builds the tokenizer of the model instead, as its `tokenizer.json` and llama.cpp do. The sentencepiece tokenizers follow transformers.

A GGUF file has no `lstrip` or `rstrip` flags. Phi-3's `tokenizer.json` strips the whitespace after its chat tokens (`<|user|>`, `<|end|>`, ...): there, the newline after `<|user|>` is not a token. From a GGUF file it is a token, as in transformers' GGUF tokenizer and in llama.cpp.

The tiny Qwen3.5 repository has the model type `qwen3_5_text`, for which transformers picks `Qwen2Tokenizer` and the Qwen2 regex. Qwen/Qwen3.5-0.8B has the model type `qwen3_5`, for which transformers loads `tokenizer.json` as it is, with the Qwen3.5 regex (letters and marks in one piece). The GGUF files say `qwen35`, so `GgufTokenizerTest` compares with the `tokenizer.json` loaded as it is.

## Tests

`GgufCausalLMTest` loads three GGUF files of the tiny Qwen3.5 model (`--outtype f32`, `bf16` and `q8_0` of `convert_hf_to_gguf.py`, `--no-mtp`). The tiny model has 2 key heads and 4 value heads, so the tests cover the tiled order. The F32 file has the weights of the tiny model, so its reference data is that of the ONNX export. For BF16 and Q8_0, `tools/causal_lm/make_models.py gguf-references` rounds the matrices as the file holds them (and BF16 further to float16, as Unity converts it), and runs transformers. The Q8_0 file covers the packed path on the GPU and the CPU. The tiny model's rows of 32 values are too short for the K-quants; the layer tests of sentis-utils cover those. The files are in [julienkay/tiny-random-Qwen3_5ForCausalLM-GGUF](https://huggingface.co/julienkay/tiny-random-Qwen3_5ForCausalLM-GGUF). `GgufLlamaTest` loads the F32, BF16 and Q8_0 files of the tiny Llama (SmolLM2 tokenizer), Phi-3 and Qwen3 models. The tiny Qwen3 model has 2 layers and the tokenizer of Qwen3.5 (`make_models.py tiny --arch qwen3`). `make_models.py gguf-references --f32` writes the reference data of all three files. The files are in [julienkay/tiny-random-LlamaForCausalLM-GGUF](https://huggingface.co/julienkay/tiny-random-LlamaForCausalLM-GGUF), [julienkay/tiny-random-Phi3ForCausalLM-GGUF](https://huggingface.co/julienkay/tiny-random-Phi3ForCausalLM-GGUF) and [julienkay/tiny-random-Qwen3ForCausalLM-GGUF](https://huggingface.co/julienkay/tiny-random-Qwen3ForCausalLM-GGUF). In the cache test, a BF16 file has a float16 cache, so the cached keys and values are rounded to float16 and the tolerance is larger.

`GgufTokenizerTest` compares the tokenizer of the Q8_0 files of the tiny Qwen3.5, Llama and Phi-3 models with the tokenizer of their Hugging Face files on fixed and random texts, and checks the special tokens and the chat template.

AI Model Hub does not serve GGUF files yet: the tests read them from the folders in the environment variables `DOJI_TINY_QWEN3_5_GGUF`, `DOJI_TINY_LLAMA_GGUF`, `DOJI_TINY_PHI3_GGUF` and `DOJI_TINY_QWEN3_GGUF`, and are ignored without them.

## Limits

- Only the architectures `llama`, `qwen3`, `phi3` and `qwen35`. Not `qwen2` (no `Qwen2ForCausalLM` class yet).
- The LongRoPE factors of the long-context Phi-3 models (`rope_factors_long`) and the rope scaling types of `rope.scaling.type` are not supported, and throw.
- Byte-level BPE tokenizers only with the pre-tokenizers `qwen35`, `qwen2` and `smollm`. Sentencepiece BPE tokenizers. Not the Unigram tokenizers (`t5`) or the other pre-tokenizers (`llama-bpe` of Llama 3, for example).
- Q4_0, Q4_1, Q5_0, Q5_1, Q2_K and Q3_K tensors load as float16. Other types (the IQ types, for example) are not converted.
- The prefill kernels of BlockQuant are naive: they do not tile the input rows.
- A tied head of a float16 table (no `output.weight`, embedding not in a packed type) is a second, transposed float16 copy of the table.
