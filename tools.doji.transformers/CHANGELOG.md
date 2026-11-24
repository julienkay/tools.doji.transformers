# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

- Text generation with decoder-only language models exported to plain ONNX (the input and output names of optimum): `AutoModelForCausalLM`, `LlamaForCausalLM` (for example SmolLM2) and `Phi3ForCausalLM`.
- `FromPretrained` loads `model.sentis` when the repository has it, and `model.onnx` otherwise.
- `PreTrainedModel.GenerateAsync` and `Generate`, like transformers' `generate` with greedy decoding or sampling: a KV cache on the GPU (`DynamicCache`), the logits processors for repetition penalty, minimum new tokens, temperature, top-k, top-p and min-p, and stopping at the maximum length, the maximum time or an end-of-sequence token.
- `PreTrainedModel.ForwardAsync` reads a prompt longer than `PreTrainedModel.PrefillChunkSize` (256 tokens) in chunks, like the chunked prefill of vLLM and llama.cpp, so one run of the model holds the activations and logits of at most 256 tokens. `DynamicCache.Clear` empties a cache.
- On a model loaded from a `.sentis` file, the worker frees each key and value of the cache after the layer that reads it, and the cache takes the new ones from the worker without a copy, so the KV cache is in GPU memory only once.
- A model with float16 weights on the GPU backend keeps its KV cache in float16, like transformers and vLLM do for a float16 model, so the cache takes half the memory. `PreTrainedModel.CacheDataType` and `DynamicCache.DataType` give the data type of the cache. The indexer of `Cache` returns `Tensor`, which is a `Tensor<short>` in a float16 cache.
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