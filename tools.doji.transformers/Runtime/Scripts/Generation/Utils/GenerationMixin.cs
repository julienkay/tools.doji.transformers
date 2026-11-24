using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using Unity.Profiling;
using Random = System.Random;

namespace Doji.AI.Transformers {

    public abstract partial class PreTrainedModel {

        /// <summary>
        /// Generates token ids that continue <paramref name="inputIds"/>, with greedy decoding or multinomial sampling,
        /// as transformers' <c>generate</c> does for one sequence.
        /// </summary>
        /// <param name="inputIds">The prompt, for example from <see cref="PreTrainedTokenizerBase.Encode"/>.</param>
        /// <param name="generationConfig">The settings. Null uses <see cref="GenerationConfig"/>.</param>
        /// <param name="streamer">Gets the prompt, then each new token, for example a <see cref="TextStreamer"/>.</param>
        /// <param name="random">The random numbers for sampling. Pass a seeded one for repeatable results.</param>
        /// <returns>The prompt followed by the new tokens.</returns>
        public Task<int[]> GenerateAsync(IReadOnlyList<int> inputIds, GenerationConfig generationConfig = null, BaseStreamer streamer = null,
            Random random = null, CancellationToken cancellationToken = default) {
            return GenerateAsync(inputIds, generationConfig, streamer, random, false, cancellationToken);
        }

        /// <inheritdoc cref="GenerateAsync(IReadOnlyList{int}, GenerationConfig, BaseStreamer, Random, CancellationToken)"/>
        public int[] Generate(IReadOnlyList<int> inputIds, GenerationConfig generationConfig = null, BaseStreamer streamer = null, Random random = null) {
            return SyncTask.Run(GenerateAsync(inputIds, generationConfig, streamer, random, true, default));
        }

        private async Task<int[]> GenerateAsync(IReadOnlyList<int> inputIds, GenerationConfig generationConfig, BaseStreamer streamer,
            Random random, bool sync, CancellationToken cancellationToken) {
            if (inputIds == null || inputIds.Count == 0) {
                throw new ArgumentException("The prompt has no tokens.", nameof(inputIds));
            }
            var config = generationConfig ?? GenerationConfig;
            if (config.NumBeams != 1) {
                throw new NotSupportedException("Beam search is not supported. Set `num_beams` to 1.");
            }
            random ??= new Random();
            var ids = new List<int>(inputIds);
            int maxLength = config.MaxNewTokens.HasValue ? ids.Count + config.MaxNewTokens.Value : config.MaxLength;
            LogitsProcessorList processors = GetLogitsProcessor(config, ids.Count);
            StoppingCriteriaList stoppingCriteria = GetStoppingCriteria(config, maxLength);

            streamer?.Put(inputIds);
            if (_worker.backendType == BackendType.GPUCompute && config.UseCache) {
                return await GenerateOnGpuAsync(ids, maxLength, config, stoppingCriteria, streamer, random, sync, cancellationToken);
            }
            var cache = CreateCache();
            try {
                while (ids.Count < maxLength) {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!config.UseCache) {
                        cache.Dispose();
                        cache = CreateCache();
                    }
                    // Like prepare_inputs_for_generation: run only the tokens that are not in the cache.
                    var newIds = ids.GetRange(cache.GetSeqLength(), ids.Count - cache.GetSeqLength());
                    float[] scores = await ForwardAsync(newIds, cache, sync);
                    processors.Process(ids, scores);
                    int next = config.DoSample ? Multinomial(scores, random.NextDouble()) : ArgMax(scores);
                    ids.Add(next);
                    streamer?.Put(new[] { next });
                    if (stoppingCriteria.IsDone(ids)) {
                        break;
                    }
                }
            } finally {
                cache.Dispose();
                streamer?.End();
            }
            return ids.ToArray();
        }

        private GpuSampler _sampler;

        // The main-thread work of GenerateAsync in a frame, and its parts.
        private static readonly ProfilerMarker FrameMarker = new("Doji.Generate.Frame");
        private static readonly ProfilerMarker StepMarker = new("Doji.Generate.Step");
        private static readonly ProfilerMarker SampleMarker = new("Doji.Generate.Sample");
        private static readonly ProfilerMarker TokensMarker = new("Doji.Generate.Tokens");
        private GenerationPacing _pacing = new GenerationPacing();

        /// <summary>
        /// How much work <see cref="GenerateAsync(IReadOnlyList{int}, GenerationConfig, BaseStreamer, Random, CancellationToken)"/>
        /// does per frame on the GPU backend. It is read when a generation starts.
        /// </summary>
        public GenerationPacing Pacing {
            get => _pacing;
            set => _pacing = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <summary>
        /// Generates with the logits processors and the sampling on the GPU (<see cref="GpuSampler"/>): each step
        /// runs on the token that the step before picked, still on the GPU. The CPU reads each token back
        /// asynchronously, for the streamer and the stopping criteria. <see cref="StepPacer"/> sets how many steps
        /// are scheduled per frame. When a stopping criterion is met, the steps already scheduled are thrown away.
        /// </summary>
        private async Task<int[]> GenerateOnGpuAsync(List<int> ids, int maxLength, GenerationConfig config, StoppingCriteriaList stoppingCriteria,
            BaseStreamer streamer, Random random, bool sync, CancellationToken cancellationToken) {
            int maxNewTokens = maxLength - ids.Count;
            var cache = CreateCache();
            var pending = new Queue<(Task<int> token, int frame)>();
            System.Collections.IEnumerator step = null;
            try {
                if (maxNewTokens <= 0) {
                    return ids.ToArray();
                }
                cancellationToken.ThrowIfCancellationRequested();
                var logits = Prefill(ids, cache);
                var settings = new GpuSampler.Key(config);
                int vocabSize = logits.shape[-1];
                if (_sampler == null || !_sampler.Settings.Equals(settings) || _sampler.VocabSize != vocabSize) {
                    _sampler?.Dispose();
                    _sampler = null;
                    _sampler = new GpuSampler(settings, vocabSize);
                }
                _sampler.Reset(ids);
                int sampled = 0;
                void Sample(Tensor<float> scores) {
                    using var marker = SampleMarker.Auto();
                    _sampler.Schedule(scores, sampled, config.DoSample ? random.NextDouble() : 0);
                    sampled++;
                    var token = sync ? Task.FromResult(ReadToken(_sampler.Token)) : ReadTokenAsync(_sampler.Token);
                    pending.Enqueue((token, sync ? 0 : UnityEngine.Time.frameCount));
                }
                Sample(logits);
                var pacer = sync ? null : new StepPacer(Pacing);
                var clock = new System.Diagnostics.Stopwatch();
                bool FrameBudgetUsed() => clock.Elapsed.TotalMilliseconds >= pacer.MaxMillisecondsPerFrame;
                while (true) {
                    cancellationToken.ThrowIfCancellationRequested();
                    int latency = -1;
                    int steps = 0;
                    int stepsPerFrame = pacer?.StepsPerFrame ?? 1;
                    int inFlight = pacer?.InFlight ?? 1;
                    using (FrameMarker.Auto()) {
                        clock.Restart();
                        while (true) {
                            if (step == null) {
                                bool start = steps < stepsPerFrame && sampled < maxNewTokens && pending.Count < inFlight
                                    && (steps == 0 || !FrameBudgetUsed());
                                if (!start) {
                                    break;
                                }
                                if (pacer == null || !pacer.SpreadStepsOverFrames) {
                                    using (StepMarker.Auto()) {
                                        Run(_sampler.Token, cache);
                                    }
                                    Sample((Tensor<float>)_worker.PeekOutput(_logits));
                                    steps++;
                                    continue;
                                }
                                step = RunInParts(_sampler.Token, cache, FrameBudgetUsed);
                            }
                            // A step spread over frames: schedule its layers until the frame budget is used.
                            bool more;
                            using (StepMarker.Auto()) {
                                more = step.MoveNext();
                            }
                            if (more) {
                                break;
                            }
                            step = null;
                            Sample((Tensor<float>)_worker.PeekOutput(_logits));
                            steps++;
                        }
                        using (TokensMarker.Auto()) {
                            while (pending.Count > 0 && pending.Peek().token.IsCompleted) {
                                var (token, frame) = pending.Dequeue();
                                // The token is done, so this does not wait.
                                int next = token.GetAwaiter().GetResult();
                                latency = sync ? 0 : UnityEngine.Time.frameCount - frame;
                                ids.Add(next);
                                streamer?.Put(new[] { next });
                                if (stoppingCriteria.IsDone(ids)) {
                                    return ids.ToArray();
                                }
                            }
                        }
                        if (pending.Count == 0 && sampled >= maxNewTokens) {
                            return ids.ToArray();
                        }
                    }
                    if (!sync) {
                        await UnityEngine.Awaitable.NextFrameAsync(cancellationToken);
                        pacer.EndFrame(UnityEngine.Time.unscaledDeltaTime, latency, limited: steps == stepsPerFrame);
                    }
                }
            } finally {
                // A step still spread over frames: its inputs are freed, and the cache is thrown away.
                (step as IDisposable)?.Dispose();
                cache.Dispose();
                streamer?.End();
            }
        }

        private static int ReadToken(Tensor<int> token) {
            using var cpu = token.ReadbackAndClone();
            return cpu[0];
        }

        /// <summary>
        /// Requests the readback now, so that it reads this token before the next step overwrites it.
        /// </summary>
        private static async Task<int> ReadTokenAsync(Tensor<int> token) {
            using var cpu = await token.ReadbackAndCloneAsync();
            return cpu[0];
        }

        /// <summary>
        /// The logits processors that the config asks for, in the order of transformers.
        /// </summary>
        public LogitsProcessorList GetLogitsProcessor(GenerationConfig config, int promptLength) => CreateLogitsProcessor(config, promptLength);

        internal static LogitsProcessorList CreateLogitsProcessor(GenerationConfig config, int promptLength) {
            var processors = new LogitsProcessorList();
            if (config.RepetitionPenalty != 1f) {
                processors.Add(new RepetitionPenaltyLogitsProcessor(config.RepetitionPenalty));
            }
            if (config.MinNewTokens > 0 && config.EosTokenId != null) {
                processors.Add(new MinNewTokensLengthLogitsProcessor(promptLength, config.MinNewTokens.Value, config.EosTokenId));
            }
            if (config.DoSample) {
                if (config.Temperature != 1f) {
                    processors.Add(new TemperatureLogitsWarper(config.Temperature));
                }
                if (config.TopK > 0) {
                    processors.Add(new TopKLogitsWarper(config.TopK));
                }
                if (config.TopP < 1f) {
                    processors.Add(new TopPLogitsWarper(config.TopP));
                }
                if (config.MinP != null) {
                    processors.Add(new MinPLogitsWarper(config.MinP.Value));
                }
            }
            return processors;
        }

        /// <summary>
        /// The stopping criteria that the config asks for.
        /// </summary>
        public StoppingCriteriaList GetStoppingCriteria(GenerationConfig config, int maxLength) {
            var criteria = new StoppingCriteriaList { new MaxLengthCriteria(maxLength) };
            if (config.MaxTime != null) {
                criteria.Add(new MaxTimeCriteria(config.MaxTime.Value));
            }
            if (config.EosTokenId != null && config.EosTokenId.Length > 0) {
                criteria.Add(new EosTokenCriteria(config.EosTokenId));
            }
            return criteria;
        }

        internal static int ArgMax(float[] scores) {
            int best = 0;
            for (int i = 1; i < scores.Length; i++) {
                if (scores[i] > scores[best]) {
                    best = i;
                }
            }
            return best;
        }

        /// <summary>
        /// Draws one token from the softmax of the scores, with a uniform random number in [0, 1).
        /// </summary>
        internal static int Multinomial(float[] scores, double draw) {
            float[] probabilities = TopPLogitsWarper.Softmax(scores);
            for (int i = 0; i < probabilities.Length; i++) {
                draw -= probabilities[i];
                if (draw < 0) {
                    return i;
                }
            }
            return ArgMax(scores);
        }
    }
}
