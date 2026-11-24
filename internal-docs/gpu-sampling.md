# Sampling on the GPU

`GenerateAsync` and `Generate` pick each token on the GPU when the model runs on `BackendType.GPUCompute` with `use_cache` on. The CPU path (`ForwardAsync`, the `LogitsProcessor` classes, `Multinomial`) stays for the CPU backend and for `use_cache = false`.

## Why

A decode step reads one token and gives the logits of the next one. On the CPU path each step waits for the logits to come back before the next step can start:

- `AsyncGPUReadback` completes about 3 frames after its request in a Player, and `GL.Flush` or awaiting does not make it faster. So the speed is at most one token per 3 frames.
- A blocking readback (`ReadbackAndClone`) stops the main thread until the GPU finishes the step.

With the sampling on the GPU, the token that a step picks stays on the GPU as the `input_ids` of the next step. The CPU reads each token back asynchronously (4 bytes) only for the streamer and the stopping criteria, and it schedules the next steps before the readback arrives.

## How

- `GpuSampler` is a small model built with `FunctionalGraph` and run by its own `Worker`. It reads the logits output of the language model where the language model's worker left them. It is built once per set of settings (`GpuSampler.Key`) and vocabulary size, and kept on the model.
- The graph does the logits processors in the order of transformers: repetition penalty, min new tokens, temperature, top-k, top-p, min-p. Then argmax, or multinomial sampling.
- The repetition penalty reads a presence vector (1 for each token in the sequence). The sampler outputs the presence vector with the new token added, so the CPU never sends the sequence again.
- Min new tokens reads a flag (1 while fewer than `min_new_tokens` tokens are generated), set by the CPU per step.
- Multinomial sampling reads a uniform random number from the CPU's `System.Random`, per step. The token is the first one in vocabulary order whose cumulative probability is above the number. This is the inverse transform that `Multinomial` does on the CPU, so a seeded `Random` gives the same draws on both paths. Float rounding of the cumulative sum (float32 on the GPU, float32 summed into a double on the CPU) can change the token only when the number falls within rounding of a boundary.
- The token is copied out of the sampler's worker (`CopyOutput`), so the next run of either worker can reuse its own buffers.
- `GenerateOnGpuAsync` keeps up to `TokensInFlight` (4) steps scheduled ahead of the token that the CPU last read. When a stopping criterion is met, the steps already scheduled are thrown away with the cache. `Generate` (sync) keeps one step in flight and reads each token with a blocking readback.

## Differences from the CPU processors

- Top-k uses `Functional.TopK` and keeps the scores at or above the k-th highest. Ties at the threshold keep all tied tokens, as `TopKLogitsWarper` does.
- Top-p sorts with `Functional.TopK` over the whole vocabulary (or over the top-k values when top-k runs first, as the other scores are minus infinity). It keeps the scores at or above the score of the last kept token. When tokens tie at that score, the GPU keeps all of them, while `TopPLogitsWarper` keeps them by its sort order.
- `min_tokens_to_keep` is 1, as in the CPU processors that `GetLogitsProcessor` makes.

## Inference Engine workaround

The graph never uses `Where(Less(x, constant), ...)`. That subgraph starts the atan2 pattern of `ContractSubExpressionPass`, and its matcher throws a `NullReferenceException` in `SubgraphMatcher.IsSmallConstant`: the constants of a `FunctionalGraph` have no `partialTensor` when the pass runs. `Greater(constant, x)` does the same and does not match the pattern.
