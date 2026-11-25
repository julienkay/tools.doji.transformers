using System;
using System.IO;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// Loads the tiny Qwen3.5 model from GGUF files of llama.cpp and compares it with transformers.
    /// convert_hf_to_gguf.py of llama.cpp makes the files from the torch folder of the tiny model (--no-mtp, --outtype
    /// f32, bf16 or q8_0). The F32 file has the weights of the tiny model, so its reference data is that of the tiny
    /// model. For BF16 and Q8_0, tools/causal_lm/make_models.py gguf-references rounds the weights as the file and
    /// Unity do.
    /// </summary>
    [TestFixture("model-f32.gguf", "tiny_qwen3_5", BackendType.GPUCompute)]
    [TestFixture("model-f32.gguf", "tiny_qwen3_5", BackendType.CPU)]
    [TestFixture("model-bf16.gguf", "tiny_qwen3_5_gguf_bf16", BackendType.GPUCompute)]
    [TestFixture("model-bf16.gguf", "tiny_qwen3_5_gguf_bf16", BackendType.CPU)]
    [TestFixture("model-q8_0.gguf", "tiny_qwen3_5_gguf_q8_0", BackendType.GPUCompute)]
    [TestFixture("model-q8_0.gguf", "tiny_qwen3_5_gguf_q8_0", BackendType.CPU)]
    public class GgufCausalLMTest {

        private readonly string _fileName, _references;
        private readonly BackendType _backend;
        private PreTrainedModel _model;
        private int[] _prompt, _greedy;

        public GgufCausalLMTest(string fileName, string references, BackendType backend) {
            _fileName = fileName;
            _references = references;
            _backend = backend;
        }

        [OneTimeSetUp]
        public void Load() {
            string directory = Environment.GetEnvironmentVariable(TinyModels.Qwen3_5GgufVariable);
            string path = directory == null ? null : Path.Combine(directory, _fileName);
            if (path == null || !File.Exists(path)) {
                Assert.Ignore($"Set {TinyModels.Qwen3_5GgufVariable} to the folder of the tiny Qwen3.5 GGUF files ({_fileName}).");
            }
            _model = AutoModelForCausalLM.FromGguf(path, _backend);
            _prompt = TestUtils.LoadInts("tiny_qwen3_5_prompt_ids");
            _greedy = TestUtils.LoadInts(_references + "_greedy");
        }

        [OneTimeTearDown]
        public void TearDown() {
            _model?.Dispose();
        }

        /// <summary>
        /// The config from the metadata of the file is the config of the tiny model, with the end-of-sequence token
        /// of the converter (&lt;|im_end|&gt;).
        /// </summary>
        [Test]
        public void Config_IsTheConfigOfTheTinyModel() {
            Assert.That(_model, Is.TypeOf<Qwen3_5ForCausalLM>());
            var config = _model.Config;
            Assert.That(config.VocabSize, Is.EqualTo(248320));
            Assert.That(config.HiddenSize, Is.EqualTo(32));
            Assert.That(config.NumHiddenLayers, Is.EqualTo(4));
            Assert.That(config.NumAttentionHeads, Is.EqualTo(4));
            Assert.That(config.NumKeyValueHeads, Is.EqualTo(2));
            Assert.That(config.HeadDim, Is.EqualTo(16));
            Assert.That(config.LayerTypes, Is.EqualTo(new[] { "linear_attention", "linear_attention", "linear_attention", "full_attention" }));
            Assert.That(config.LinearNumKeyHeads, Is.EqualTo(2));
            Assert.That(config.LinearNumValueHeads, Is.EqualTo(4));
            Assert.That(config.LinearKeyHeadDim, Is.EqualTo(16));
            Assert.That(config.LinearValueHeadDim, Is.EqualTo(8));
            Assert.That(config.LinearConvKernelDim, Is.EqualTo(4));
            Assert.That(_model.GenerationConfig.EosTokenId, Is.EqualTo(new[] { 248046 }));
            Assert.That(_model.GenerationConfig.PadTokenId, Is.EqualTo(248044));
        }

        [Test]
        public void Logits_MatchTransformers() {
            using var cache = _model.CreateCache();
            float[] logits = _model.Forward(_prompt, cache);
            float[] expected = TestUtils.LoadFloats(_references + "_last_logits");
            TestUtils.AssertClose(expected, logits[..expected.Length], 1e-3f, "last logits");
        }

        [Test]
        public void Logits_WithCache_MatchWithoutCache() {
            using var cache = _model.CreateCache();
            _model.Forward(_prompt[..5], cache);
            float[] cached = _model.Forward(_prompt[5..], cache);
            using var fresh = _model.CreateCache();
            float[] full = _model.Forward(_prompt, fresh);
            TestUtils.AssertClose(full, cached, 1e-3f, "logits");
        }

        [Test]
        public void GreedyGeneration_MatchesTransformers() {
            int[] sequence = _model.Generate(_prompt, new GenerationConfig { MaxNewTokens = _greedy.Length, EosTokenId = new[] { 248046 } });
            Assert.That(sequence[_prompt.Length..], Is.EqualTo(_greedy));
        }
    }
}
