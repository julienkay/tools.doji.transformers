# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

- Text generation with decoder-only language models exported to plain ONNX (the input and output names of optimum): `AutoModelForCausalLM`, `LlamaForCausalLM` (for example SmolLM2) and `Phi3ForCausalLM`.
- `FromPretrained` loads `model.sentis` when the repository has it, and `model.onnx` otherwise.
- `PreTrainedModel.GenerateAsync` and `Generate`, like transformers' `generate` with greedy decoding or sampling: a KV cache on the GPU (`DynamicCache`), the logits processors for repetition penalty, minimum new tokens, temperature, top-k, top-p and min-p, and stopping at the maximum length, the maximum time or an end-of-sequence token.
- On the GPU backend, `GenerateAsync` and `Generate` run the logits processors and the sampling on the GPU. Each step reads the token that the step before picked, still on the GPU, so it does not wait for a readback of the logits. `GenerateAsync` reads the tokens back asynchronously for the streamer and the stopping criteria, and schedules steps ahead of them: a small model runs several steps per frame. On the CPU backend, and with `use_cache` off, the logits come back to the CPU on each step.
- `PreTrainedModel.Pacing` (`GenerationPacing`) limits the work of `GenerateAsync` per frame on the GPU backend: `MaxStepsPerFrame` and `MaxMillisecondsPerFrame`, the main-thread time for scheduling steps. Lower them to leave more of each frame to the rest of the app. With `SpreadStepsOverFrames`, one step can take several frames, so a model whose step takes longer than a frame keeps the frames short.
- `PreTrainedModel.ForwardAsync` reads a prompt longer than `PreTrainedModel.PrefillChunkSize` (256 tokens) in chunks, like the chunked prefill of vLLM and llama.cpp, so one run of the model holds the activations and logits of at most 256 tokens. `DynamicCache.Clear` empties a cache.
- On a model loaded from a `.sentis` file, the worker frees each key and value of the cache after the layer that reads it, and the cache takes the new ones from the worker without a copy, so the KV cache is in GPU memory only once.
- A model with float16 weights on the GPU backend keeps its KV cache in float16, like transformers and vLLM do for a float16 model, so the cache takes half the memory. `PreTrainedModel.CacheDataType` and `DynamicCache.DataType` give the data type of the cache. The indexer of `Cache` returns `Tensor`, which is a `Tensor<short>` in a float16 cache.
- `Qwen3_5ForCausalLM`, for Qwen3.5. Its layers mix full attention with the linear attention of Gated DeltaNet, which needs `LinearAttention.GatedDeltaRule` of tools.doji.sentis-utils. `AutoModelForCausalLM` loads the text model of a `Qwen3_5ForConditionalGeneration` checkpoint with it, as transformers does.
- `AutoModelForCausalLM.FromGguf` and `Qwen3_5ForCausalLM.FromGguf` load a Qwen3.5 model from a GGUF file of llama.cpp (architecture `qwen35`), with no ONNX export and no `.sentis` file. The config comes from the metadata of the file and the special tokens from its tokenizer metadata. The tensors stay in the file until the worker needs them, and go to the GPU one at a time. Linear and embedding weights of type Q8_0, Q4_K, Q5_K or Q6_K stay quantized on the device (`BlockQuant` of sentis-utils), so they take the memory of their blocks only. F32 tensors stay float32. F16, BF16 and the other supported quantized types become float16; quantized tensors that do log a warning. It reads the files of older and newer llama.cpp converters. Depends on `tools.doji.formats.gguf`.
- `AutoTokenizer.FromGguf` loads the tokenizer of a GGUF file from its metadata: the vocabulary, the merges, the added and special tokens, and the chat template. It supports byte-level BPE tokenizers with the pre-tokenizers `qwen35`, `qwen2` and `smollm`, and sentencepiece BPE tokenizers such as the tokenizer of Phi-3.
- `FromGguf` loads GGUF files of the architectures `llama` (for example SmolLM2), `qwen3` and `phi3`, with `LlamaForCausalLM`, `Qwen3ForCausalLM` and `Phi3ForCausalLM`. `AutoModelForCausalLM.FromGguf` picks the class from the architecture of the file.
- `Qwen3ForCausalLM`, for Qwen3. `AutoModelForCausalLM` loads it for a `Qwen3ForCausalLM` checkpoint.
- `DynamicCache(config)`, like transformers' `DynamicCache(config=...)`: for each linear attention layer, it holds the state of the causal convolution and the state of the gated delta rule, which each step replaces. `DynamicCache.IsLinearAttention` tells which layers these are.
- `PretrainedConfig.GetTextConfig`, `TextConfig`, `ModelType`, `LayerTypes` and the `Linear*` sizes of linear attention layers.
- `GenerationConfig.FromModelConfig`, like transformers' `GenerationConfig.from_model_config`. A model without `generation_config.json` takes its special tokens from `config.json`, and from its `text_config` where the top level has none.
- `ApplyChatTemplate` adds the thinking block of the Qwen3.5 template to the generation prompt: empty, or open when `enableThinking` is true.
- `TextStreamer`, which passes on the text as it is generated.
- `TokenizersBackend`, the tokenizer of transformers v5: it runs the `Tokenizer` of `tools.doji.tokenizers`, a C# port of the Hugging Face tokenizers library, and loads any `tokenizer.json` (BPE, Unigram, WordPiece and WordLevel models).
- `Qwen2Tokenizer`, and truncation, padding, pairs and batches as transformers v5 does them.
- `ApplyChatTemplate` for ChatML and Phi-3 chat templates.
- `AutoTokenizer.FromPretrainedAsync`, which picks the tokenizer class from the model type and the tokenizer config as transformers v5 does.
- A text generation sample: a chat with SmolLM2-360M-Instruct or SmolLM2-135M-Instruct, which replaces the Phi-3 sample. SmolLM2-1.7B-Instruct and Phi-3-mini are optional models of the sample: turn them on in AI Model Hub to download them. The SmolLM2 models come as float16 `.sentis` files (julienkay/SmolLM2-*-Instruct-sentis-float16), which give the answers of float32 at half the memory and load without conversion. Phi-3-mini comes as a uint8 `.sentis` file (3.8 GB, julienkay/Phi-3-mini-4k-instruct-sentis-uint8) and as a float16 `.sentis` file (7.6 GB, julienkay/Phi-3-mini-4k-instruct-sentis-float16). The sample ships a Model Hub manifest asset, so Model Hub downloads the models after the import. Its UI is a UI Toolkit document: chat bubbles that stream the reply, a Stop button, and the sampling settings of the model card. It works with any render pipeline.
- Tests that compare a tiny random Llama model with onnxruntime and transformers, and the tokenizers with the official ones on many edge cases.

### Changed

- Models and tokenizers are delivered by AI Model Hub (`tools.doji.model-hub`). List the models in the project's AI Model Hub manifest.
- Depends on `com.unity.ai.inference` 2.6.1 and Unity 6000.3.
- All tokenizers work as in transformers v5: `GPT2Tokenizer`, `CLIPTokenizer` (renamed from `ClipTokenizer`), `LlamaTokenizer` and `Qwen2Tokenizer` build their pipeline on `TokenizersBackend` from `tokenizer.json`, `vocab.json` and `merges.txt`, or `tokenizer.model`. As in v5, `CLIPTokenizer` no longer runs ftfy and `LlamaTokenizer` is not legacy by default.
- The special tokens of `PreTrainedTokenizerBase` are strings, and `TokenizerConfig` reads them as the `AddedToken` of `tools.doji.tokenizers`.
- Depends on `tools.doji.tokenizers` instead of `tools.doji.sentencepiece` and its DLLs.

### Fixed

- On the GPU backend, the main thread stopped at the start of each step of `GenerateAsync` until the GPU had done the work of the step before, because the model read the sequence length back from the GPU. Models loaded from a file and models created from a `Model` in memory now keep it on the CPU.

### Removed

- Loading models from `Resources` and `StreamingAssets`.
- The slow tokenizers (`PreTrainedTokenizer`, `BasicTokenizer`, `Trie`, ftfy's `FixText`) and `PreTrainedTokenizerFast`, as transformers v5 removed them.
- The unfinished beam search, the cache classes other than `DynamicCache`, `Kwargs` and `ModelOutput`.

## [0.1.6] - 2026-02-17

### Changed

- Updated Sentis dependency to v2.5.0

## [0.1.5] - 2025-04-10

### Fixed

- Improved protocol buffer based tokenizer.model handling, which fixes issues preventing package to be used at runtime

## [0.1.4] - 2025-04-04

## [0.1.3] - 2025-04-04

### Changed

- Changed the dependency version of tools.doji.sentencepiece from 0.0.0 to 0.1.0

## [0.1.2] - 2025-04-01

## [0.1.1] - 2024-08-16

### Fixed

- Fixed mising dependency in package manifest

## [0.1.0] - 2024-08-16

### Added

- LlamaTokenizer
- GPT2Tokenizer
- A (preliminary) implementation for running Phi-3 Mini models

## [0.0.2] - 2024-04-10

### Fixed

- Fixed issue where ClipTokenizer initialization fails with ArgumentOutOfRangeException

## [0.0.1] - 2024-03-24

### Added

- Custom JsonConverter for tokenizer inputs
- Custom JsonConverter for vocab files

## [0.0.0] - 2024-01-29

- Initial Release
- Supports CLIP tokenization