using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// The tokenizer of a GGUF file (<see cref="AutoTokenizer.FromGguf"/>) against the tokenizer.json of the same
    /// model: the tiny Qwen3.5 model, which has the tokenizer of Qwen/Qwen3.5-0.8B.
    /// </summary>
    public class GgufTokenizerTest {

        private PreTrainedTokenizerBase _fromGguf, _reference;

        [OneTimeSetUp]
        public void Load() {
            string directory = Environment.GetEnvironmentVariable(TinyModels.Qwen3_5GgufVariable);
            string path = directory == null ? null : Path.Combine(directory, "model-q8_0.gguf");
            if (path == null || !File.Exists(path)) {
                Assert.Ignore($"Set {TinyModels.Qwen3_5GgufVariable} to the folder of the tiny Qwen3.5 GGUF files.");
            }
            _fromGguf = AutoTokenizer.FromGguf(path);
            // The tiny repository has the model type qwen3_5_text, for which transformers picks Qwen2Tokenizer and its
            // pre-tokenizer. Qwen/Qwen3.5-0.8B has the model type qwen3_5, for which transformers loads tokenizer.json
            // as it is. The GGUF files keep the tokenizer of the original model, so the reference is the second.
            _reference = SyncTask.Run(AutoTokenizer.FromPretrainedAsync(ModelRepository.Open(TinyModels.Qwen3_5), "", "qwen3_5"));
        }

        private static IEnumerable<string> Texts() {
            yield return "What is the capital of France?";
            yield return "  Hello,\tworld!\n\n  It's 2026: 3.14159 × 10⁶ — naïve café 東京 😀 <think>\n</think>";
            yield return "def f(x):\n    return x ** 2  # comment\r\n";
            yield return "I'VE been there, we'll see; they'd've";
            yield return "ﬁ Å Å ́e";
            yield return "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
            var random = new Random(3);
            for (int i = 0; i < 200; i++) {
                yield return new string(Enumerable.Range(0, random.Next(1, 40)).Select(_ => (char)random.Next(32, 0x3000)).ToArray());
            }
        }

        [Test]
        public void Encode_IsTheEncodingOfTokenizerJson() {
            foreach (string text in Texts()) {
                int[] expected = _reference.Encode(text).InputIds.ToArray();
                int[] actual = _fromGguf.Encode(text).InputIds.ToArray();
                Assert.That(actual, Is.EqualTo(expected), text);
                Assert.That(_fromGguf.Decode(actual), Is.EqualTo(_reference.Decode(expected)), text);
            }
        }

        [Test]
        public void SpecialTokens_AreThoseOfTheConverter() {
            // convert_hf_to_gguf.py sets the end-of-sequence token to <|im_end|>, the end of a chat turn.
            Assert.That(_fromGguf.EosToken, Is.EqualTo("<|im_end|>"));
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
