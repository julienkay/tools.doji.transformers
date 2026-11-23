using Newtonsoft.Json.Linq;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// Compares the tokenizers with the tokenizers of transformers v5 on many edge cases: the class AutoTokenizer picks,
    /// whitespace, contractions, numbers, punctuation, code, scripts, emoji, special tokens, each token decoded alone,
    /// pairs, batches, truncation, padding and chat templates.
    /// The reference data comes from tools/tokenizers/make_tokenizer_references.py; the companion project's
    /// model-hub manifest pins the tokenizer files at the same revisions.
    /// </summary>
    public class TokenizerReferenceTest {

        /// <summary>The reference files in Resources/tokenizers.</summary>
        private static readonly string[] Names = { "smollm2", "gpt2", "qwen2_5", "phi3", "phi3_spm", "clip" };

        private static readonly Dictionary<string, JObject> References = Names.ToDictionary(n => n, n => JObject.Parse(TestUtils.LoadText($"tokenizers/{n}")));
        private static readonly Dictionary<string, PreTrainedTokenizerBase> Tokenizers = new();

        private static PreTrainedTokenizerBase Tokenizer(string name) {
            if (!Tokenizers.TryGetValue(name, out var tokenizer)) {
                JObject reference = References[name];
                var files = ModelRepository.Open((string)reference["repo"]);
                string subfolder = (string)reference["subfolder"];
                if (name.EndsWith("_spm")) {
                    // Loaded from tokenizer.model alone, as transformers does without tokenizer.json.
                    var config = TokenizerConfig.Deserialize(SyncTask.Run(files.ReadTextAsync(subfolder + "tokenizer_config.json")));
                    tokenizer = LlamaTokenizer.FromSentencePiece(SyncTask.Run(files.ReadFileAsync(subfolder + "tokenizer.model")), config);
                } else {
                    // The subfolders of the test repository have no config.json, so the model type comes from the reference.
                    tokenizer = SyncTask.Run(AutoTokenizer.FromPretrainedAsync(files, subfolder, (string)reference["model_type"]));
                }
                Tokenizers[name] = tokenizer;
            }
            return tokenizer;
        }

        private static string Label(string text) {
            string escaped = text.Replace("\\", "\\\\").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
            return escaped.Length > 40 ? escaped[..40] + "..." : escaped;
        }

        private static IEnumerable<TestCaseData> Items(string key, Func<string, int, JToken, string> label) {
            foreach (string name in Names) {
                JArray items = (JArray)References[name][key];
                for (int i = 0; i < items.Count; i++) {
                    yield return new TestCaseData(name, i).SetName($"{name} {label(name, i, items[i])}");
                }
            }
        }

        public static IEnumerable<TestCaseData> Cases() => Items("cases", (_, i, c) => $"{i:D2} \"{Label((string)c["text"])}\"");
        public static IEnumerable<TestCaseData> Calls() => Items("calls", (_, i, c) => $"call {i:D2} {Label(c["kwargs"].ToString(Newtonsoft.Json.Formatting.None))}");
        public static IEnumerable<TestCaseData> Chats() => Items("chats", (_, i, _) => $"chat {i}");

        [TestCaseSource(nameof(Names))]
        public void Class_MatchesTransformers(string name) {
            Assert.That(Tokenizer(name).GetType().Name, Is.EqualTo((string)References[name]["class"]));
        }

        [TestCaseSource(nameof(Cases))]
        public void Encode_MatchesTransformers(string name, int index) {
            JToken c = References[name]["cases"][index];
            int[] ids = Tokenizer(name).Encode((string)c["text"]).InputIds.ToArray();
            Assert.That(ids, Is.EqualTo(c["ids"].ToObject<int[]>()));
        }

        [TestCaseSource(nameof(Cases))]
        public void Decode_MatchesTransformers(string name, int index) {
            JToken c = References[name]["cases"][index];
            var ids = c["ids"].ToObject<List<int>>();
            var tokenizer = Tokenizer(name);
            Assert.That(tokenizer.Decode(ids), Is.EqualTo((string)c["decoded"]), "skipSpecialTokens: false");
            Assert.That(tokenizer.Decode(ids, skipSpecialTokens: true), Is.EqualTo((string)c["decoded_skip_special"]), "skipSpecialTokens: true");
        }

        /// <summary>
        /// Each token decoded alone, which splits characters that span several tokens.
        /// </summary>
        [TestCaseSource(nameof(Cases))]
        public void DecodeEachToken_MatchesTransformers(string name, int index) {
            JToken c = References[name]["cases"][index];
            var tokenizer = Tokenizer(name);
            string[] each = c["ids"].Select(id => tokenizer.Decode(new List<int> { (int)id })).ToArray();
            Assert.That(each, Is.EqualTo(c["decoded_each"].ToObject<string[]>()));
        }

        /// <summary>A call of the tokenizer with pairs, batches, truncation or padding.</summary>
        [TestCaseSource(nameof(Calls))]
        public void Call_MatchesTransformers(string name, int index) {
            JToken call = References[name]["calls"][index];
            JObject kwargs = (JObject)call["kwargs"];
            Input Text(JToken json) => json switch {
                null => null,
                JArray list => new BatchInput(list.ToObject<List<string>>()),
                _ => (string)json,
            };
            Truncation truncation = kwargs["truncation"] switch {
                null => Truncation.None,
                JValue { Type: JTokenType.Boolean } value => (bool)value ? Truncation.LongestFirst : Truncation.None,
                JToken value => (string)value switch {
                    "only_first" => Truncation.OnlyFirst,
                    "only_second" => Truncation.OnlySecond,
                    _ => Truncation.LongestFirst,
                },
            };
            Padding padding = kwargs["padding"] switch {
                null => Padding.None,
                JValue { Type: JTokenType.Boolean } value => (bool)value ? Padding.Longest : Padding.None,
                JToken value => (string)value == "max_length" ? Padding.MaxLength : Padding.Longest,
            };
            Encoding Encode() => Tokenizer(name).Encode(
                Text(kwargs["text"]),
                Text(kwargs["text_pair"]),
                addSpecialTokens: (bool?)kwargs["add_special_tokens"] ?? true,
                padding: padding,
                truncation: truncation,
                maxLength: (int?)kwargs["max_length"],
                padToMultipleOf: (int?)kwargs["pad_to_multiple_of"],
                returnTokenTypeIds: (bool?)kwargs["return_token_type_ids"],
                returnSpecialTokensMask: (bool?)kwargs["return_special_tokens_mask"] ?? false);

            if (call["error"] != null) {
                Assert.That(() => Encode(), Throws.Exception, $"transformers raises {call["error"]}");
                return;
            }
            Encoding encoding = Encode();
            JObject expected = (JObject)call["output"];
            Assert.That(encoding.Keys.OrderBy(k => k), Is.EqualTo(expected.Properties().Select(p => p.Name).OrderBy(k => k)));
            foreach (var property in expected.Properties()) {
                Assert.That(JToken.FromObject(encoding[property.Name]).ToString(Newtonsoft.Json.Formatting.None),
                    Is.EqualTo(property.Value.ToString(Newtonsoft.Json.Formatting.None)), property.Name);
            }
        }

        [TestCaseSource(nameof(Chats))]
        public void ApplyChatTemplate_MatchesTransformers(string name, int index) {
            JToken chat = References[name]["chats"][index];
            var messages = chat["messages"].Select(m => new ChatMessage((string)m["role"], (string)m["content"]));
            string text = Tokenizer(name).ApplyChatTemplate(messages, (bool)chat["add_generation_prompt"]);
            Assert.That(text, Is.EqualTo((string)chat["text"]));
        }
    }
}
