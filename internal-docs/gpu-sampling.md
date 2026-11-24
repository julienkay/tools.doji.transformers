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
- `GenerateOnGpuAsync` schedules steps ahead of the token that the CPU last read. Each readback is requested right after its sampler step, so it reads that step's token even though the next step overwrites the buffer. When a stopping criterion is met, the steps already scheduled are thrown away with the cache. `Generate` (sync) keeps one step in flight and reads each token with a blocking readback.

## Steps per frame

`StepPacer` sets how many steps `GenerateAsync` schedules per frame, within the limits of `PreTrainedModel.Pacing` (`GenerationPacing`):

- It doubles the steps per frame (TCP slow start) until a frame is slow, then adds one step per frame. It halves the steps when a frame takes more than 1.5 times the shortest frame of the generation, or when a token comes back more than `ReadbackFrames + 1` frames after its step (the GPU has more work queued than it does in a frame). It grows only in a frame that scheduled all its steps.
- `GenerationPacing.MaxMillisecondsPerFrame` (default 4) caps the main-thread time for scheduling steps in a frame. The first step of a frame is always scheduled. For a small model, scheduling on the CPU (Inference Engine's per-layer dispatch) is the limit, not the GPU, so this budget sets the speed.
- `GenerationPacing.MaxStepsPerFrame` (default 32) caps the steps per frame.
- The in-flight window is `StepsPerFrame * (ReadbackFrames + 1)`, at least 4: a readback takes about 3 frames.
- A model whose one step takes longer than a frame runs one step per frame by default, and the frames get longer.
- `GenerationPacing.SpreadStepsOverFrames` spreads each step over frames instead: `RunInParts` schedules the layers of a step with `ScheduleInParts` of `tools.doji.sentis-utils`, and a part ends when the frame's `MaxMillisecondsPerFrame` is used. The sampler runs in the frame that finishes the step. `ScheduleInParts` puts the worker on an external command buffer, which it executes once per part, not after every layer. So on the GPU backend the model owns a `CommandBuffer` from the start, and `Run` executes it after each full run.
- The budget is main-thread time, and the GPU work of a part follows the CPU time only roughly. For a model whose GPU time per step is much larger than its CPU time, a part can still carry more GPU work than a frame. A smaller budget helps there.
- If generation stops (a stopping criterion, or cancellation) while a step is spread, the step's enumerator is disposed and the cache is thrown away. The next run of the worker starts fresh.
- `GenerationPacing` has no counterpart in transformers. It changes when the tokens come, not which tokens.

## Differences from the CPU processors

- Top-k uses `Functional.TopK` and keeps the scores at or above the k-th highest. Ties at the threshold keep all tied tokens, as `TopKLogitsWarper` does.
- Top-p sorts with `Functional.TopK` over the whole vocabulary (or over the top-k values when top-k runs first, as the other scores are minus infinity). It keeps the scores at or above the score of the last kept token. When tokens tie at that score, the GPU keeps all of them, while `TopPLogitsWarper` keeps them by its sort order.
- `min_tokens_to_keep` is 1, as in the CPU processors that `GetLogitsProcessor` makes.

## Inference Engine workaround

The graph never uses `Where(Less(x, constant), ...)`. That subgraph starts the atan2 pattern of `ContractSubExpressionPass`, and its matcher throws a `NullReferenceException` in `SubgraphMatcher.IsSmallConstant`: the constants of a `FunctionalGraph` have no `partialTensor` when the pass runs. `Greater(constant, x)` does the same and does not match the pattern.
