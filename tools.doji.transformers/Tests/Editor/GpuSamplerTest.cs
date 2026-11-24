using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// The GPU sampler picks the tokens of the logits processors and the sampling on the CPU, for the same draws.
    /// </summary>
    public class GpuSamplerTest {

        private const int Vocab = 1000;
        private static readonly double[] Draws = { 0.01, 0.2, 0.45, 0.5, 0.77, 0.93, 0.999 };

        private static float[] RandomLogits(int seed, float scale = 3f, float offset = 0f) {
            var random = new System.Random(seed);
            return Enumerable.Range(0, Vocab).Select(_ => offset + scale * (float)(random.NextDouble() * 2 - 1)).ToArray();
        }

        private static IEnumerable<TestCaseData> Configs() {
            yield return new TestCaseData(new GenerationConfig()).SetName("Greedy");
            yield return new TestCaseData(new GenerationConfig { RepetitionPenalty = 1.3f }).SetName("Greedy_RepetitionPenalty");
            yield return new TestCaseData(new GenerationConfig { DoSample = true, TopK = 0 }).SetName("Sample");
            yield return new TestCaseData(new GenerationConfig { DoSample = true, TopK = 0, Temperature = 0.7f }).SetName("Sample_Temperature");
            yield return new TestCaseData(new GenerationConfig { DoSample = true, TopK = 20 }).SetName("Sample_TopK");
            yield return new TestCaseData(new GenerationConfig { DoSample = true, TopK = 0, TopP = 0.9f }).SetName("Sample_TopP");
            yield return new TestCaseData(new GenerationConfig { DoSample = true, TopK = 50, TopP = 0.8f }).SetName("Sample_TopK_TopP");
            yield return new TestCaseData(new GenerationConfig { DoSample = true, TopK = 0, MinP = 0.05f }).SetName("Sample_MinP");
            yield return new TestCaseData(new GenerationConfig {
                DoSample = true, Temperature = 0.6f, TopK = 40, TopP = 0.95f, MinP = 0.02f, RepetitionPenalty = 1.2f
            }).SetName("Sample_All");
        }

        [TestCaseSource(nameof(Configs))]
        public void PicksTheTokensOfTheCpu(GenerationConfig config) {
            int[] prompt = { 3, 17, 256, 999, 42 };
            for (int seed = 0; seed < 4; seed++) {
                float[] scores = RandomLogits(seed);
                // The prompt holds the best tokens, so that the repetition penalty changes the choice.
                foreach (int id in prompt) {
                    scores[id] = 3.5f + id % 3 * 0.1f;
                }
                foreach (double draw in config.DoSample ? Draws : new[] { 0.0 }) {
                    float[] cpu = (float[])scores.Clone();
                    PreTrainedModel.CreateLogitsProcessor(config, prompt.Length).Process(prompt, cpu);
                    int expected = config.DoSample ? PreTrainedModel.Multinomial(cpu, draw) : PreTrainedModel.ArgMax(cpu);
                    Assert.That(Sample(config, prompt, scores, 0, draw), Is.EqualTo(expected), $"seed {seed}, draw {draw}");
                }
            }
        }

        [Test]
        public void MinNewTokens_BlocksTheEosToken() {
            float[] scores = RandomLogits(7);
            int eos = PreTrainedModel.ArgMax(scores);
            var config = new GenerationConfig { MinNewTokens = 2, EosTokenId = new[] { eos } };
            int[] prompt = { 1 };
            Assert.That(Sample(config, prompt, scores, generated: 1, draw: 0), Is.Not.EqualTo(eos));
            Assert.That(Sample(config, prompt, scores, generated: 2, draw: 0), Is.EqualTo(eos));
        }

        [Test]
        public void RepetitionPenalty_AppliesToTheTokensItPicked() {
            float[] scores = RandomLogits(11, scale: 1f, offset: 2f);
            int best = PreTrainedModel.ArgMax(scores);
            var config = new GenerationConfig { RepetitionPenalty = 100f };
            using var sampler = new GpuSampler(new GpuSampler.Key(config), Vocab);
            sampler.Reset(new[] { (best + 1) % Vocab });
            using var logits = new Tensor<float>(new TensorShape(1, 1, Vocab), scores);
            sampler.Schedule(logits, 0, 0);
            Assert.That(Read(sampler.Token), Is.EqualTo(best));
            sampler.Schedule(logits, 1, 0);
            int second = Read(sampler.Token);
            Assert.That(second, Is.Not.EqualTo(best));
            float[] cpu = (float[])scores.Clone();
            new RepetitionPenaltyLogitsProcessor(100f).Process(new[] { (best + 1) % Vocab, best }, cpu);
            Assert.That(second, Is.EqualTo(PreTrainedModel.ArgMax(cpu)));
        }

        /// <summary>
        /// Runs the sampler on logits of two positions, of which the last scores the token.
        /// </summary>
        private static int Sample(GenerationConfig config, int[] prompt, float[] scores, int generated, double draw) {
            using var sampler = new GpuSampler(new GpuSampler.Key(config), Vocab);
            sampler.Reset(prompt);
            var data = new float[2 * Vocab];
            scores.CopyTo(data, Vocab);
            using var logits = new Tensor<float>(new TensorShape(1, 2, Vocab), data);
            sampler.Schedule(logits, generated, draw);
            return Read(sampler.Token);
        }

        private static int Read(Tensor<int> token) {
            using var cpu = token.ReadbackAndClone();
            Assert.That(cpu.shape, Is.EqualTo(new TensorShape(1, 1)));
            return cpu[0];
        }
    }
}
