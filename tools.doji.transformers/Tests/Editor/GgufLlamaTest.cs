using System;
using System.IO;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// Loads the tiny Llama, Phi-3 and Qwen3 models from GGUF files of llama.cpp (architectures <c>llama</c>,
    /// <c>phi3</c> and <c>qwen3</c>) and compares them with transformers. convert_hf_to_gguf.py of llama.cpp makes the
    /// files from the torch folder of each tiny model (--outtype f32, bf16 or q8_0), and
    /// tools/causal_lm/make_models.py gguf-references --f32 writes the reference data: transformers with the weights
    /// rounded as the file and Unity do.
    /// </summary>
    [TestFixture(TinyModels.LlamaGgufVariable, "tiny_llama", "f32", BackendType.GPUCompute)]
    [TestFixture(TinyModels.LlamaGgufVariable, "tiny_llama", "q8_0", BackendType.GPUCompute)]
    [TestFixture(TinyModels.LlamaGgufVariable, "tiny_llama", "q8_0", BackendType.CPU)]
    [TestFixture(TinyModels.LlamaGgufVariable, "tiny_llama", "bf16", BackendType.GPUCompute)]
    [TestFixture(TinyModels.Phi3GgufVariable, "tiny_phi3", "f32", BackendType.GPUCompute)]
    [TestFixture(TinyModels.Phi3GgufVariable, "tiny_phi3", "q8_0", BackendType.GPUCompute)]
    [TestFixture(TinyModels.Phi3GgufVariable, "tiny_phi3", "q8_0", BackendType.CPU)]
    [TestFixture(TinyModels.Phi3GgufVariable, "tiny_phi3", "bf16", BackendType.GPUCompute)]
    [TestFixture(TinyModels.Qwen3GgufVariable, "tiny_qwen3", "f32", BackendType.GPUCompute)]
    [TestFixture(TinyModels.Qwen3GgufVariable, "tiny_qwen3", "q8_0", BackendType.GPUCompute)]
    [TestFixture(TinyModels.Qwen3GgufVariable, "tiny_qwen3", "q8_0", BackendType.CPU)]
    [TestFixture(TinyModels.Qwen3GgufVariable, "tiny_qwen3", "bf16", BackendType.GPUCompute)]
    public class GgufLlamaTest {

        private readonly string _variable, _prefix, _outType;
        private readonly BackendType _backend;
        private PreTrainedModel _model;
        private int[] _prompt, _greedy;

        public GgufLlamaTest(string variable, string prefix, string outType, BackendType backend) {
            _variable = variable;
            _prefix = prefix;
            _outType = outType;
            _backend = backend;
        }

        [OneTimeSetUp]
        public void Load() {
            string directory = Environment.GetEnvironmentVariable(_variable);
            string path = directory == null ? null : Path.Combine(directory, $"model-{_outType}.gguf");
            if (path == null || !File.Exists(path)) {
                Assert.Ignore($"Set {_variable} to the folder of the GGUF files of {_prefix} (model-{_outType}.gguf).");
            }
            _model = AutoModelForCausalLM.FromGguf(path, _backend);
            _prompt = TestUtils.LoadInts(_prefix + "_prompt_ids");
            _greedy = TestUtils.LoadInts($"{_prefix}_gguf_{_outType}_greedy");
        }

        [OneTimeTearDown]
        public void TearDown() {
            _model?.Dispose();
        }

        /// <summary>The config from the metadata of the file is the config of the tiny model.</summary>
        [Test]
        public void Config_IsTheConfigOfTheTinyModel() {
            var config = _model.Config;
            Assert.That(config.HiddenSize, Is.EqualTo(32));
            Assert.That(config.NumHiddenLayers, Is.EqualTo(2));
            Assert.That(config.NumAttentionHeads, Is.EqualTo(4));
            switch (_prefix) {
                case "tiny_llama":
                    Assert.That(_model, Is.TypeOf<LlamaForCausalLM>());
                    Assert.That(config.VocabSize, Is.EqualTo(49152));
                    Assert.That(config.NumKeyValueHeads, Is.EqualTo(2));
                    Assert.That(config.HeadDim, Is.EqualTo(8));
                    break;
                case "tiny_phi3":
                    Assert.That(_model, Is.TypeOf<Phi3ForCausalLM>());
                    Assert.That(config.VocabSize, Is.EqualTo(32064));
                    Assert.That(config.NumKeyValueHeads, Is.EqualTo(4));
                    Assert.That(config.HeadDim, Is.EqualTo(8));
                    break;
                default:
                    Assert.That(_model, Is.TypeOf<Qwen3ForCausalLM>());
                    Assert.That(config.VocabSize, Is.EqualTo(248320));
                    Assert.That(config.NumKeyValueHeads, Is.EqualTo(2));
                    Assert.That(config.HeadDim, Is.EqualTo(16));
                    break;
            }
        }

        [Test]
        public void Logits_MatchTransformers() {
            using var cache = _model.CreateCache();
            float[] logits = _model.Forward(_prompt, cache);
            float[] expected = TestUtils.LoadFloats($"{_prefix}_gguf_{_outType}_last_logits");
            TestUtils.AssertClose(expected, logits[..expected.Length], 1e-3f, "last logits");
        }

        [Test]
        public void Logits_WithCache_MatchWithoutCache() {
            using var cache = _model.CreateCache();
            _model.Forward(_prompt[..5], cache);
            float[] cached = _model.Forward(_prompt[5..], cache);
            using var fresh = _model.CreateCache();
            float[] full = _model.Forward(_prompt, fresh);
            // A model with float16 weights has a float16 cache: the cached keys and values are rounded to float16,
            // the keys and values of the same run are not. The large weights of the tiny models spread the logits.
            float tolerance = _model.CacheDataType == DataType.Short ? 3e-2f : 1e-3f;
            TestUtils.AssertClose(full, cached, tolerance, "logits");
        }

        [Test]
        public void GreedyGeneration_MatchesTransformers() {
            int[] sequence = _model.Generate(_prompt, new GenerationConfig { MaxNewTokens = _greedy.Length });
            Assert.That(sequence[_prompt.Length..], Is.EqualTo(_greedy));
        }
    }
}
