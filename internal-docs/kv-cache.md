# KV cache and chunked prefill (internal notes)

How text generation keeps memory low on long prompts. Built on the sentis-utils features described in `internal-docs/` of tools.doji.sentis-utils.

## KV cache

### What it does

To generate a token, every attention layer needs the keys and values of all earlier tokens. The KV cache keeps them, so each new token only runs the model on itself instead of on the whole text again. `DynamicCache` holds one key tensor and one value tensor per layer, on the GPU, and they grow by the new tokens on every run.

### Similar approach

transformers' **`DynamicCache`** is the same idea and the default in transformers: the cache grows with `torch.cat` on every step. transformers also has **`StaticCache`** (`cache_implementation="static"`, `max_cache_len`), which reserves the full length once and writes new tokens into it.

### How it works with Inference Engine

The ONNX export (optimum) gives the model `past_key_values.*` inputs and `present.*` outputs. Inside, the model concatenates past and new keys and values. So each run:

1. **Give the cache in.** `WorkerUtils.TryGiveInput` gives each past key and value to the worker. The worker's storage then owns them and frees each one right after the layer that reads it.
2. **Take the new cache out.** `WorkerUtils.TakeOutput` takes each `present.*` output out of the worker without a copy, and `DynamicCache.Update` stores it.

So the cache exists once in GPU memory, not once in the cache and once more in the worker. This needs the worker from `ModelFile.CreateWorker` ([FittedModelStorage](../../tools.doji.sentis-utils/internal-docs/fitted-model-storage.md)).

### float16 cache

A model with float16 weights on the GPU keeps its cache in float16 (`PreTrainedModel.CacheDataType`), as transformers and vLLM do for float16 models. `ModelFile.UseHalfPrecision` makes the `past_key_values.*` inputs and `present.*` outputs float16, with casts next to the layers that use them ([HalfPrecision](../../tools.doji.sentis-utils/internal-docs/half-precision.md)). The cache then takes half the memory. The `Cache` indexer returns `Tensor`: a `Tensor<short>` in a float16 cache.

## Chunked prefill

### What it does

The first run reads the whole prompt. For a long prompt, the activations and the logits of all its tokens at once can be bigger than the cache. `ForwardAsync` runs a long prompt in chunks of `PrefillChunkSize` (256) tokens, each after the cache of the ones before. The result is the same as one run; only the peak memory is lower.

### Similar approach

**vLLM's chunked prefill** and **llama.cpp's batch size** (`n_batch` / `n_ubatch`) split a long prompt the same way.

### How it works

It's a plain loop over `Run(chunk, cache)` with the cache passed along. Positions and the attention mask cover past plus new tokens. If a run throws, the cache is cleared, because the worker may already have freed the keys and values it was given.

## Known limit on D3D12: fragmentation

On D3D12, Unity places every GPU buffer under 32 MiB in 64 MiB heaps and does not give the heaps back after the buffers are released. During chunked prefill, the KV-sized tensors (the cache itself, its float32 copies inside the layers) get a new, bigger size every chunk. They leave holes in those heaps that the next, bigger tensors can't use. VRAM then grows with the prompt length beyond the live tensors: on a 12 GB GPU, Phi-3 float16 reached the limit at about 2048 tokens.

- **D3D11** doesn't show this. On D3D11 the held VRAM grows only by the cache, and the same GPU ran 3072 tokens. D3D11 was also faster for every model tested.
- A **static (preallocated) cache**, like transformers' `StaticCache`, would stop the cache from growing. It needs a layer that writes new tokens into place and an attention kernel that reads a cache with a row stride. Not built yet.
- A repro for the Unity heap behavior is in `G:\Unity\BugRepros\`.
