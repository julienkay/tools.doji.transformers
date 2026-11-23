using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

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
                    int next = config.DoSample ? Multinomial(scores, random) : ArgMax(scores);
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

        /// <summary>
        /// The logits processors that the config asks for, in the order of transformers.
        /// </summary>
        public LogitsProcessorList GetLogitsProcessor(GenerationConfig config, int promptLength) {
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

        private static int ArgMax(float[] scores) {
            int best = 0;
            for (int i = 1; i < scores.Length; i++) {
                if (scores[i] > scores[best]) {
                    best = i;
                }
            }
            return best;
        }

        /// <summary>
        /// Draws one token from the softmax of the scores.
        /// </summary>
        private static int Multinomial(float[] scores, Random random) {
            float[] probabilities = TopPLogitsWarper.Softmax(scores);
            double draw = random.NextDouble();
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
