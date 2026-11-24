using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using Random = System.Random;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Picks the next token on the GPU: the logits processors of a <see cref="GenerationConfig"/>, then greedy
    /// decoding or multinomial sampling, as a small model that reads the logits where the language model left them.
    /// The token stays on the GPU as the input of the next step, so a step does not wait for a readback.
    /// </summary>
    /// <remarks>
    /// It does what <see cref="PreTrainedModel.GetLogitsProcessor"/> and the CPU sampling of
    /// <see cref="PreTrainedModel.GenerateAsync(IReadOnlyList{int}, GenerationConfig, BaseStreamer, Random, System.Threading.CancellationToken)"/>
    /// do, in the order of transformers. Sampling draws the token from a random number of the CPU, so a seeded
    /// <see cref="Random"/> gives the tokens of the CPU path.
    /// </remarks>
    internal sealed class GpuSampler : IDisposable {

        private readonly Worker _worker;
        private readonly int _vocabSize;
        private readonly bool _penalty, _minNewTokens, _sample;
        private readonly int _logitsInput, _presenceInput = -1, _flagInput = -1, _drawInput = -1;
        private readonly int _tokenOutput, _presenceOutput = -1;
        private Tensor _presence, _token;

        /// <summary>The number of tokens of the vocabulary.</summary>
        public int VocabSize => _vocabSize;

        /// <summary>The settings this sampler was made for.</summary>
        public Key Settings { get; }

        /// <summary>The last token, shape (1, 1), on the GPU. The next <see cref="Schedule"/> overwrites it.</summary>
        public Tensor<int> Token => (Tensor<int>)_token;

        public GpuSampler(Key settings, int vocabSize) {
            Settings = settings;
            _vocabSize = vocabSize;
            _penalty = settings.RepetitionPenalty != 1f;
            _minNewTokens = settings.EosTokenId.Length > 0 && settings.MinNewTokens > 0;
            _sample = settings.DoSample;

            var graph = new FunctionalGraph();
            var inputs = new List<FunctionalTensor>();
            FunctionalTensor Input<T>(TensorShape shape) where T : unmanaged {
                var input = graph.AddInput<T>(shape);
                inputs.Add(input);
                return input;
            }
            var logits = graph.AddInput<float>(new DynamicTensorShape(1, -1, vocabSize));
            inputs.Add(logits);
            _logitsInput = 0;
            var minusInfinity = Functional.Constant(float.NegativeInfinity);
            // The scores of the next token: the logits of the last position.
            var scores = logits[0, ^1, ..];

            FunctionalTensor presence = null;
            if (_penalty) {
                _presenceInput = inputs.Count;
                presence = Input<float>(new TensorShape(vocabSize));
                // RepetitionPenaltyLogitsProcessor.
                float penalty = settings.RepetitionPenalty;
                // Greater(0, x), not Less(x, 0): Where(Less(x, constant)) starts the atan2 pattern of Inference Engine's
                // ContractSubExpressionPass, whose matcher fails on the constants of a FunctionalGraph.
                var penalized = Functional.Where(Functional.Greater(Functional.Constant(0f), scores), scores * penalty, scores / penalty);
                scores = Functional.Where(Functional.Greater(presence, Functional.Constant(0.5f)), penalized, scores);
            }
            if (_minNewTokens) {
                // MinNewTokensLengthLogitsProcessor: the flag is 1 while fewer than min_new_tokens are generated.
                _flagInput = inputs.Count;
                var flag = Input<float>(new TensorShape(1));
                var eos = new float[vocabSize];
                foreach (int id in settings.EosTokenId) {
                    eos[id] = 1f;
                }
                var blocked = Functional.Greater(Functional.Constant(new TensorShape(vocabSize), eos) * flag, Functional.Constant(0.5f));
                scores = Functional.Where(blocked, minusInfinity, scores);
            }

            FunctionalTensor token;
            if (!_sample) {
                token = Functional.ArgMax(scores, 0, keepdim: true);
            } else {
                _drawInput = inputs.Count;
                var draw = Input<float>(new TensorShape(1));
                if (settings.Temperature != 1f) {
                    scores = scores / settings.Temperature;
                }
                FunctionalTensor sorted = null;
                int topK = Math.Max(settings.TopK, 1);
                if (settings.TopK > 0 && topK < vocabSize) {
                    // TopKLogitsWarper: scores below the k-th highest go to minus infinity.
                    sorted = Functional.TopK(scores, topK, 0)[0];
                    var threshold = sorted.Narrow(0, topK - 1, 1);
                    scores = Functional.Where(Functional.Less(scores, threshold), minusInfinity, scores);
                }
                if (settings.TopP < 1f) {
                    // TopPLogitsWarper: a token goes when the more likely tokens already reach top_p. After top-k,
                    // the k highest scores are all the tokens left, so their softmax is the softmax of all scores.
                    sorted ??= Functional.TopK(scores, vocabSize, 0)[0];
                    var probabilities = Functional.Softmax(sorted, 0);
                    var before = Functional.CumSum(probabilities, 0) - probabilities;
                    var kept = Functional.ReduceSum(Functional.Less(before, Functional.Constant(settings.TopP)).Int(), 0, keepdim: true);
                    kept = Functional.Max(kept, Functional.Constant(new[] { 1 }));
                    var cutoff = sorted.Gather(0, kept - 1);
                    scores = Functional.Where(Functional.Less(scores, cutoff), minusInfinity, scores);
                }
                if (settings.MinP != null) {
                    // MinPLogitsWarper: the most likely token always stays.
                    var probabilities = Functional.Softmax(scores, 0);
                    var threshold = Functional.ReduceMax(probabilities, 0, keepdim: true) * settings.MinP.Value;
                    scores = Functional.Where(Functional.Less(probabilities, threshold), minusInfinity, scores);
                }
                // Multinomial sampling by the inverse of the cumulative distribution, in vocabulary order: the
                // first token whose cumulative probability is above the draw. Rounding can leave none: then the
                // most likely token, as on the CPU.
                var cumulative = Functional.CumSum(Functional.Softmax(scores, 0), 0);
                var below = Functional.ReduceSum(Functional.LessEqual(cumulative, draw).Int(), 0, keepdim: true);
                token = Functional.Where(Functional.Greater(Functional.Constant(new[] { vocabSize }), below), below,
                    Functional.ArgMax(scores, 0, keepdim: true));
            }
            token = token.Reshape(new[] { 1, 1 });

            var outputs = new List<FunctionalTensor> { token };
            _tokenOutput = 0;
            if (_penalty) {
                // The token joins the tokens that the repetition penalty applies to.
                var chosen = Functional.Equals(Functional.ARange(vocabSize), token.Reshape(new[] { 1 })).Float();
                outputs.Add(Functional.Max(presence, chosen));
                _presenceOutput = 1;
            }
            Model model = graph.Compile(outputs.ToArray());
            _worker = new Worker(model, BackendType.GPUCompute);
        }

        /// <summary>
        /// Sets the tokens that the repetition penalty applies to: the prompt.
        /// </summary>
        public void Reset(IReadOnlyList<int> promptIds) {
            if (!_penalty) {
                return;
            }
            var presence = new float[_vocabSize];
            foreach (int id in promptIds) {
                presence[id] = 1f;
            }
            _presence?.Dispose();
            _presence = new Tensor<float>(new TensorShape(_vocabSize), presence);
        }

        /// <summary>
        /// Schedules the choice of the next token from <paramref name="logits"/>, shape (1, sequence, vocabulary).
        /// </summary>
        /// <param name="generated">The tokens generated before this one, for min_new_tokens.</param>
        /// <param name="draw">A uniform random number in [0, 1), for sampling.</param>
        public void Schedule(Tensor<float> logits, int generated, double draw) {
            _worker.SetInput(_logitsInput, logits);
            if (_penalty) {
                _worker.SetInput(_presenceInput, _presence);
            }
            using var flag = _minNewTokens ? new Tensor<float>(new TensorShape(1), new[] { generated < Settings.MinNewTokens ? 1f : 0f }) : null;
            using var random = _sample ? new Tensor<float>(new TensorShape(1), new[] { (float)draw }) : null;
            if (flag != null) {
                _worker.SetInput(_flagInput, flag);
            }
            if (random != null) {
                _worker.SetInput(_drawInput, random);
            }
            _worker.Schedule();
            // Copies, so that the next run of either model can reuse its own buffers.
            _worker.CopyOutput(_tokenOutput, ref _token);
            if (_penalty) {
                _worker.CopyOutput(_presenceOutput, ref _presence);
            }
        }

        public void Dispose() {
            _worker.Dispose();
            _presence?.Dispose();
            _token?.Dispose();
        }

        /// <summary>
        /// The settings of a <see cref="GenerationConfig"/> that change the sampler model.
        /// </summary>
        internal sealed class Key : IEquatable<Key> {

            public bool DoSample { get; }
            public float RepetitionPenalty { get; }
            /// <summary>0 when no end-of-sequence token is blocked.</summary>
            public int MinNewTokens { get; }
            public int[] EosTokenId { get; }
            public float Temperature { get; }
            public int TopK { get; }
            public float TopP { get; }
            public float? MinP { get; }

            public Key(GenerationConfig config) {
                DoSample = config.DoSample;
                RepetitionPenalty = config.RepetitionPenalty;
                EosTokenId = config.MinNewTokens > 0 && config.EosTokenId != null ? config.EosTokenId.ToArray() : Array.Empty<int>();
                MinNewTokens = EosTokenId.Length > 0 ? config.MinNewTokens.Value : 0;
                Temperature = DoSample ? config.Temperature : 1f;
                TopK = DoSample ? config.TopK : 0;
                TopP = DoSample ? config.TopP : 1f;
                MinP = DoSample ? config.MinP : null;
            }

            public bool Equals(Key other) => other != null && DoSample == other.DoSample && RepetitionPenalty == other.RepetitionPenalty
                && MinNewTokens == other.MinNewTokens && EosTokenId.SequenceEqual(other.EosTokenId) && Temperature == other.Temperature
                && TopK == other.TopK && TopP == other.TopP && MinP == other.MinP;

            public override bool Equals(object obj) => Equals(obj as Key);

            public override int GetHashCode() => HashCode.Combine(DoSample, RepetitionPenalty, MinNewTokens, EosTokenId.Length, Temperature, TopK, TopP, MinP);
        }
    }
}
