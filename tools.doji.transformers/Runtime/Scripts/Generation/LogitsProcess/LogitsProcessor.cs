using System;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Changes the scores of the next token during generation, as transformers' logits processors do.
    /// </summary>
    /// <remarks>
    /// This package generates one sequence at a time, so a processor gets the ids so far and
    /// the scores of the next token, and changes the scores in place.
    /// </remarks>
    public abstract class LogitsProcessor {

        /// <param name="inputIds">The prompt and the tokens generated so far.</param>
        /// <param name="scores">The scores of each vocabulary token for the next position.</param>
        public abstract void Process(IReadOnlyList<int> inputIds, float[] scores);
    }

    /// <summary>
    /// Makes repeated tokens less likely: divides positive scores of tokens in <c>inputIds</c> by the penalty and
    /// multiplies negative ones.
    /// </summary>
    public class RepetitionPenaltyLogitsProcessor : LogitsProcessor {

        public float Penalty { get; }

        public RepetitionPenaltyLogitsProcessor(float penalty) {
            if (penalty <= 0) {
                throw new ArgumentException($"`penalty` has to be a strictly positive float, but is {penalty}");
            }
            Penalty = penalty;
        }

        public override void Process(IReadOnlyList<int> inputIds, float[] scores) {
            foreach (int id in inputIds.Distinct()) {
                scores[id] = scores[id] < 0 ? scores[id] * Penalty : scores[id] / Penalty;
            }
        }
    }

    /// <summary>
    /// Prevents the end-of-sequence tokens until <see cref="MinNewTokens"/> tokens are generated.
    /// </summary>
    public class MinNewTokensLengthLogitsProcessor : LogitsProcessor {

        public int PromptLength { get; }
        public int MinNewTokens { get; }
        public int[] EosTokenId { get; }

        public MinNewTokensLengthLogitsProcessor(int promptLength, int minNewTokens, int[] eosTokenId) {
            PromptLength = promptLength;
            MinNewTokens = minNewTokens;
            EosTokenId = eosTokenId;
        }

        public override void Process(IReadOnlyList<int> inputIds, float[] scores) {
            if (inputIds.Count - PromptLength < MinNewTokens) {
                foreach (int id in EosTokenId) {
                    scores[id] = float.NegativeInfinity;
                }
            }
        }
    }

    /// <summary>
    /// Divides the scores by the temperature: below 1 sharpens the distribution, above 1 flattens it.
    /// </summary>
    public class TemperatureLogitsWarper : LogitsProcessor {

        public float Temperature { get; }

        public TemperatureLogitsWarper(float temperature) {
            if (temperature <= 0) {
                throw new ArgumentException($"`temperature` has to be a strictly positive float, but is {temperature}. " +
                    "For greedy decoding, set `do_sample` to false.");
            }
            Temperature = temperature;
        }

        public override void Process(IReadOnlyList<int> inputIds, float[] scores) {
            for (int i = 0; i < scores.Length; i++) {
                scores[i] /= Temperature;
            }
        }
    }

    /// <summary>
    /// Keeps the <see cref="TopK"/> highest scores and sets the others to minus infinity.
    /// </summary>
    public class TopKLogitsWarper : LogitsProcessor {

        public int TopK { get; }

        public TopKLogitsWarper(int topK, int minTokensToKeep = 1) {
            if (topK <= 0) {
                throw new ArgumentException($"`top_k` has to be a strictly positive integer, but is {topK}");
            }
            TopK = Math.Max(topK, minTokensToKeep);
        }

        public override void Process(IReadOnlyList<int> inputIds, float[] scores) {
            if (TopK >= scores.Length) {
                return;
            }
            float[] sorted = (float[])scores.Clone();
            Array.Sort(sorted);
            float threshold = sorted[^TopK];
            for (int i = 0; i < scores.Length; i++) {
                if (scores[i] < threshold) {
                    scores[i] = float.NegativeInfinity;
                }
            }
        }
    }

    /// <summary>
    /// Keeps the smallest set of most likely tokens whose probabilities add up to <see cref="TopP"/> (nucleus sampling).
    /// </summary>
    public class TopPLogitsWarper : LogitsProcessor {

        public float TopP { get; }
        public int MinTokensToKeep { get; }

        public TopPLogitsWarper(float topP, int minTokensToKeep = 1) {
            if (topP < 0 || topP > 1) {
                throw new ArgumentException($"`top_p` has to be a float > 0 and < 1, but is {topP}");
            }
            TopP = topP;
            MinTokensToKeep = minTokensToKeep;
        }

        public override void Process(IReadOnlyList<int> inputIds, float[] scores) {
            float[] probabilities = Softmax(scores);
            int[] order = Enumerable.Range(0, scores.Length).ToArray();
            Array.Sort(probabilities.Select(p => -p).ToArray(), order);
            double cumulative = 0;
            for (int rank = 0; rank < order.Length; rank++) {
                // Remove a token when the more likely tokens already reach top_p.
                if (rank >= MinTokensToKeep && cumulative >= TopP) {
                    scores[order[rank]] = float.NegativeInfinity;
                }
                cumulative += probabilities[order[rank]];
            }
        }

        internal static float[] Softmax(float[] scores) {
            float max = scores.Max();
            var probabilities = new float[scores.Length];
            double sum = 0;
            for (int i = 0; i < scores.Length; i++) {
                probabilities[i] = MathF.Exp(scores[i] - max);
                sum += probabilities[i];
            }
            for (int i = 0; i < scores.Length; i++) {
                probabilities[i] = (float)(probabilities[i] / sum);
            }
            return probabilities;
        }
    }

    /// <summary>
    /// Keeps the tokens whose probability is at least <see cref="MinP"/> times the probability of the most likely token.
    /// </summary>
    public class MinPLogitsWarper : LogitsProcessor {

        public float MinP { get; }
        public int MinTokensToKeep { get; }

        public MinPLogitsWarper(float minP, int minTokensToKeep = 1) {
            if (minP < 0 || minP > 1) {
                throw new ArgumentException($"`min_p` has to be a float >= 0 and <= 1, but is {minP}");
            }
            MinP = minP;
            MinTokensToKeep = minTokensToKeep;
        }

        public override void Process(IReadOnlyList<int> inputIds, float[] scores) {
            float[] probabilities = TopPLogitsWarper.Softmax(scores);
            float threshold = MinP * probabilities.Max();
            int[] order = Enumerable.Range(0, scores.Length).ToArray();
            Array.Sort(probabilities.Select(p => -p).ToArray(), order);
            for (int rank = MinTokensToKeep; rank < order.Length; rank++) {
                if (probabilities[order[rank]] < threshold) {
                    scores[order[rank]] = float.NegativeInfinity;
                }
            }
        }
    }
}
