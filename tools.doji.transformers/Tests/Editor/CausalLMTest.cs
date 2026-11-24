using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// Runs the tiny models and compares them with onnxruntime and transformers.
    /// The reference data comes from tools/causal_lm/make_models.py.
    /// </summary>
    [TestFixture(TinyModels.Llama, "tiny_llama", BackendType.GPUCompute)]
    [TestFixture(TinyModels.Llama, "tiny_llama", BackendType.CPU)]
    [TestFixture(TinyModels.Phi3, "tiny_phi3", BackendType.GPUCompute)]
    [TestFixture(TinyModels.Phi3, "tiny_phi3", BackendType.CPU)]
    public class CausalLMTest {

        private readonly string _modelId, _references;
        private readonly BackendType _backend;
        private PreTrainedModel _model;
        private PreTrainedTokenizerBase _tokenizer;
        private int[] _prompt, _greedy;
        private int _eos;

        public CausalLMTest(string modelId, string references, BackendType backend) {
            _modelId = modelId;
            _references = references;
            _backend = backend;
        }

        [OneTimeSetUp]
        public void Load() {
            _model = AutoModelForCausalLM.FromPretrained(_modelId, _backend);
            _tokenizer = AutoTokenizer.FromPretrained(_modelId);
            _prompt = TestUtils.LoadInts(_references + "_prompt_ids");
            _greedy = TestUtils.LoadInts(_references + "_greedy");
            _eos = _model.GenerationConfig.EosTokenId[0];
        }

        [OneTimeTearDown]
        public void TearDown() {
            _model?.Dispose();
        }

        /// <summary>
        /// AutoModelForCausalLM picks the class of <c>architectures</c>, and AutoTokenizer the class transformers v5
        /// picks: the tokenizer_config of SmolLM2 names GPT2Tokenizer, and Phi-3 loads its tokenizer.json directly.
        /// </summary>
        [Test]
        public void AutoClasses_PickTheClassesOfTransformers() {
            if (_modelId == TinyModels.Llama) {
                Assert.That(_model, Is.TypeOf<LlamaForCausalLM>());
                Assert.That(_tokenizer, Is.TypeOf<GPT2Tokenizer>());
                Assert.That(_model.GenerationConfig.EosTokenId, Is.EqualTo(new[] { 2 }));
            } else {
                Assert.That(_model, Is.TypeOf<Phi3ForCausalLM>());
                Assert.That(_tokenizer, Is.TypeOf<TokenizersBackend>());
                Assert.That(_model.GenerationConfig.EosTokenId, Is.EqualTo(new[] { 32000 }));
            }
        }

        [Test]
        public void ChatPrompt_TokenizesLikeTransformers() {
            string chat = _tokenizer.ApplyChatTemplate(new[] { ChatMessage.User("What is the capital of France?") });
            Assert.That(_tokenizer.Encode(chat).InputIds, Is.EqualTo(_prompt));
        }

        [Test]
        public void Logits_MatchOnnxRuntime() {
            using var cache = _model.CreateCache();
            float[] logits = _model.Forward(_prompt, cache);
            float[] expected = TestUtils.LoadFloats(_references + "_last_logits");
            TestUtils.AssertClose(expected, logits[..expected.Length], 1e-3f, "last logits");
            Assert.That(cache.GetSeqLength(), Is.EqualTo(_prompt.Length));
        }

        /// <summary>
        /// The prompt in two parts, the second one after the cache of the first, gives the same logits.
        /// </summary>
        [Test]
        public void Logits_WithCache_MatchWithoutCache() {
            using var cache = _model.CreateCache();
            _model.Forward(_prompt[..5], cache);
            float[] cached = _model.Forward(_prompt[5..], cache);
            using var fresh = _model.CreateCache();
            float[] full = _model.Forward(_prompt, fresh);
            TestUtils.AssertClose(full, cached, 1e-4f, "logits");
        }

        /// <summary>
        /// Greedy decoding with the KV cache gives the tokens of transformers' <c>generate(do_sample=False)</c>.
        /// </summary>
        [Test]
        public void GreedyGeneration_MatchesTransformers() {
            int[] expected = _greedy;
            int[] sequence = _model.Generate(_prompt, new GenerationConfig { MaxNewTokens = expected.Length, EosTokenId = new[] { _eos } });
            Assert.That(sequence[..^expected.Length], Is.EqualTo(_prompt));
            Assert.That(sequence[_prompt.Length..], Is.EqualTo(expected));
        }

        [Test]
        public void GreedyGeneration_WithoutCache_IsTheSame() {
            int[] expected = _greedy;
            int[] sequence = _model.Generate(_prompt, new GenerationConfig { MaxNewTokens = 3, UseCache = false });
            Assert.That(sequence[_prompt.Length..], Is.EqualTo(expected[..3]));
        }

        [Test]
        public void Generation_StopsAtMaxLength() {
            int[] sequence = _model.Generate(_prompt, new GenerationConfig { MaxLength = _prompt.Length + 2 });
            Assert.That(sequence.Length, Is.EqualTo(_prompt.Length + 2));
        }

        [Test]
        public void Generation_StopsAtEosToken() {
            int[] expected = _greedy;
            int[] sequence = _model.Generate(_prompt, new GenerationConfig { MaxNewTokens = 8, EosTokenId = new[] { expected[1] } });
            Assert.That(sequence[_prompt.Length..], Is.EqualTo(expected[..2]));
        }

        [Test]
        public void TextStreamer_StreamsTheGeneratedText() {
            var pieces = new List<string>();
            var streamer = new TextStreamer(_tokenizer, pieces.Add, skipPrompt: true);
            int[] sequence = _model.Generate(_prompt, new GenerationConfig { MaxNewTokens = 6 }, streamer);
            Assert.That(string.Concat(pieces), Is.EqualTo(_tokenizer.Decode(sequence[_prompt.Length..].ToList(), skipSpecialTokens: true)));
        }

        [Test]
        public void Sampling_WithSeed_IsRepeatable() {
            var config = new GenerationConfig { MaxNewTokens = 4, DoSample = true, Temperature = 0.7f, TopK = 20, TopP = 0.9f };
            int[] first = _model.Generate(_prompt, config, random: new System.Random(1));
            int[] second = _model.Generate(_prompt, config, random: new System.Random(1));
            Assert.That(second, Is.EqualTo(first));
        }
    }
}
