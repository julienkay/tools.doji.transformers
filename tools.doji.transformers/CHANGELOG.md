# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

- Text generation with decoder-only language models exported to plain ONNX (the input and output names of optimum): `AutoModelForCausalLM`, `LlamaForCausalLM` (for example SmolLM2) and `Phi3ForCausalLM`.
- `PreTrainedModel.GenerateAsync` and `Generate`, like transformers' `generate` with greedy decoding or sampling: a KV cache on the GPU (`DynamicCache`), the logits processors for repetition penalty, minimum new tokens, temperature, top-k, top-p and min-p, and stopping at the maximum length, the maximum time or an end-of-sequence token.
- `TextStreamer`, which passes on the text as it is generated.
- `PreTrainedTokenizerFast`, which reads byte-level BPE `tokenizer.json` files (SmolLM2, Qwen2.5, GPT-2 and others).
- `ApplyChatTemplate` for ChatML and Phi-3 chat templates.
- `AutoTokenizer.FromPretrainedAsync`.
- A text generation sample with SmolLM2-135M-Instruct, which replaces the Phi-3 sample.
- Tests that compare a tiny random Llama model with onnxruntime and transformers, and the tokenizers with the official ones on many edge cases.

### Changed

- Models and tokenizers are delivered by AI Model Hub (`tools.doji.model-hub`). List the models in the project's AI Model Hub manifest.
- Depends on `com.unity.ai.inference` 2.6.1 and Unity 6000.3.
- `GPT2Tokenizer` no longer adds CLIP's end-of-word suffix and reads all merges.

### Removed

- Loading models from `Resources` and `StreamingAssets`.
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