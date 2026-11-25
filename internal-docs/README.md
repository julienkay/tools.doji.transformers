# Design documents

These documents describe how the package works and why. They are not work plans: a plan belongs in a pull request or an issue, not here.

| Document | Subject |
|---|---|
| [kv-cache.md](kv-cache.md) | KV cache, single-copy cache, float16 cache, chunked prefill, D3D12 fragmentation |
| [gpu-sampling.md](gpu-sampling.md) | Logits processors and sampling on the GPU, pipelined token readback, steps per frame |
| [gguf.md](gguf.md) | Loading GGUF files of llama.cpp: the graph in C#, external weights, what differs from transformers |
| [qwen3_5.md](qwen3_5.md) | Qwen3.5: linear attention states in the cache, the export with the GatedDeltaRule node, text config, thinking block of the chat template |

The Inference Engine features this package builds on (ModelFile, FittedModelStorage, HalfPrecision, attention, ...) are described in `internal-docs/` of `G:\Unity\tools.doji.sentis-utils`.
