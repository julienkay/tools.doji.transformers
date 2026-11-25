using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// The tokenizer of a GGUF file (<see cref="AutoTokenizer.FromGguf"/>) against the tokenizer of the same model
    /// from its Hugging Face files: the tiny Qwen3.5 model (byte-level BPE, pre-tokenizer qwen35), the tiny Llama
    /// model with the tokenizer of SmolLM2 (byte-level BPE, smollm) and the tiny Phi-3 model (sentencepiece BPE).
    /// </summary>
    [TestFixture(TinyModels.Qwen3_5GgufVariable, TinyModels.Qwen3_5, "<|im_end|>")]
    [TestFixture(TinyModels.LlamaGgufVariable, TinyModels.Llama, null)]
    [TestFixture(TinyModels.Phi3GgufVariable, TinyModels.Phi3, null)]
    public class GgufTokenizerTest {

        private readonly string _variable, _repository, _eosToken;
        private PreTrainedTokenizerBase _fromGguf, _reference;

        /// <param name="eosToken">The end-of-sequence token of the file, or null for the token of the reference.</param>
        public GgufTokenizerTest(string variable, string repository, string eosToken) {
            _variable = variable;
            _repository = repository;
            _eosToken = eosToken;
        }

        [OneTimeSetUp]
        public void Load() {
            string directory = Environment.GetEnvironmentVariable(_variable);
            string path = directory == null ? null : Path.Combine(directory, "model-q8_0.gguf");
            if (path == null || !File.Exists(path)) {
                Assert.Ignore($"Set {_variable} to the folder of the GGUF files of {_repository}.");
            }
            _fromGguf = AutoTokenizer.FromGguf(path);
            if (_repository == TinyModels.Qwen3_5) {
                // The tiny repository has the model type qwen3_5_text, for which transformers picks Qwen2Tokenizer and
                // its pre-tokenizer. Qwen/Qwen3.5-0.8B has the model type qwen3_5, for which transformers loads
                // tokenizer.json as it is. The GGUF files keep the tokenizer of the original model, so the reference
                // is the second.
                _reference = SyncTask.Run(AutoTokenizer.FromPretrainedAsync(ModelRepository.Open(_repository), "", "qwen3_5"));
            } else {
                _reference = AutoTokenizer.FromPretrained(_repository);
            }
        }

        private static IEnumerable<string> Texts() {
            yield return "What is the capital of France?";
            yield return "  Hello,\tworld!\n\n  It's 2026: 3.14159 × 10⁶ — naïve café 東京 😀 <think>\n</think>";
            yield return "def f(x):\n    return x ** 2  # comment\r\n";
            yield return "I'VE been there, we'll see; they'd've";
            yield return "ﬁ Å Å ́e";
            // Not the chat tokens of Phi-3: its tokenizer.json strips the whitespace after them (rstrip), and a GGUF file
            // has no such flag.
            yield return "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
            var random = new Random(3);
            for (int i = 0; i < 200; i++) {
                yield return new string(Enumerable.Range(0, random.Next(1, 40)).Select(_ => (char)random.Next(32, 0x3000)).ToArray());
            }
        }

        [Test]
        public void Encode_IsTheEncodingOfTheReference() {
            foreach (string text in Texts()) {
                int[] expected = _reference.Encode(text).InputIds.ToArray();
                int[] actual = _fromGguf.Encode(text).InputIds.ToArray();
                Assert.That(actual, Is.EqualTo(expected), text);
                Assert.That(_fromGguf.Decode(actual), Is.EqualTo(_reference.Decode(expected)), text);
            }
        }

        [Test]
        public void SpecialTokens_AreThoseOfTheConverter() {
            // convert_hf_to_gguf.py sets the end-of-sequence token of Qwen3.5 to <|im_end|>, the end of a chat turn.
            Assert.That(_fromGguf.EosToken, Is.EqualTo(_eosToken ?? _reference.EosToken));
            Assert.That(_fromGguf.BosToken, Is.EqualTo(_reference.BosToken));
            Assert.That(_fromGguf.PadToken, Is.EqualTo(_reference.PadToken));
            Assert.That(_fromGguf.Count, Is.EqualTo(_reference.Count));
        }

        [Test]
        public void ChatTemplate_IsTheTemplateOfTheModel() {
            var messages = new[] { ChatMessage.User("Hi"), ChatMessage.Assistant("<think>\nok\n</think>\n\nHello"), ChatMessage.User("And?") };
            Assert.That(_fromGguf.ApplyChatTemplate(messages), Is.EqualTo(_reference.ApplyChatTemplate(messages)));
        }
    }
}
