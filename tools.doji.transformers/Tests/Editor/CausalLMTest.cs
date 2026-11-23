using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// Runs the tiny Llama model and compares it with onnxruntime and transformers.
    /// The reference data comes from tools/causal_lm/make_models.py.
    /// </summary>
    [TestFixture(BackendType.GPUCompute)]
    [TestFixture(BackendType.CPU)]
    public class CausalLMTest {

        private readonly BackendType _backend;
        private PreTrainedModel _model;
        private PreTrainedTokenizerBase _tokenizer;
        private int[] _prompt;

        public CausalLMTest(BackendType backend) {
            _backend = backend;
        }

        [OneTimeSetUp]
        public void Load() {
            _model = AutoModelForCausalLM.FromPretrained(TinyModels.Llama, _backend);
            _tokenizer = AutoTokenizer.FromPretrained(TinyModels.Llama);
            _prompt = TestUtils.LoadInts("tiny_llama_prompt_ids");
        }

        [OneTimeTearDown]
        public void TearDown() {
            _model?.Dispose();
        }

        [Test]
        public void AutoClasses_PickLlama() {
            Assert.That(_model, Is.TypeOf<LlamaForCausalLM>());
            Assert.That(_tokenizer, Is.TypeOf<PreTrainedTokenizerFast>());
            Assert.That(_model.GenerationConfig.EosTokenId, Is.EqualTo(new[] { 2 }));
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
            float[] expected = TestUtils.LoadFloats("tiny_llama_last_logits");
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
            int[] expected = TestUtils.LoadInts("tiny_llama_greedy");
            int[] sequence = _model.Generate(_prompt, new GenerationConfig { MaxNewTokens = expected.Length, EosTokenId = new[] { 2 } });
            Assert.That(sequence[..^expected.Length], Is.EqualTo(_prompt));
            Assert.That(sequence[_prompt.Length..], Is.EqualTo(expected));
        }

        [Test]
        public void GreedyGeneration_WithoutCache_IsTheSame() {
            int[] expected = TestUtils.LoadInts("tiny_llama_greedy");
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
            int[] expected = TestUtils.LoadInts("tiny_llama_greedy");
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
