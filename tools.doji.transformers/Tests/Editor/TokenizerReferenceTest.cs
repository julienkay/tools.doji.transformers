using Newtonsoft.Json.Linq;
using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// Compares the tokenizers with the official transformers tokenizers on many edge cases: whitespace, contractions,
    /// numbers, punctuation, code, scripts, emoji, special tokens, and each token decoded alone.
    /// The reference data comes from tools/tokenizers/make_tokenizer_references.py; the companion project's
    /// model-hub manifest pins the tokenizer files at the same revisions.
    /// </summary>
    public class TokenizerReferenceTest {

        /// <summary>The reference files in Resources/tokenizers.</summary>
        private static readonly string[] Names = { "smollm2", "gpt2", "gpt2_slow", "qwen2_5", "phi3", "clip" };

        private static readonly Dictionary<string, JObject> References = Names.ToDictionary(n => n, n => JObject.Parse(TestUtils.LoadText($"tokenizers/{n}")));
        private static readonly Dictionary<string, PreTrainedTokenizerBase> Tokenizers = new();

        private static PreTrainedTokenizerBase Tokenizer(string name) {
            if (!Tokenizers.TryGetValue(name, out var tokenizer)) {
                var files = ModelRepository.Open((string)References[name]["repo"]);
                string subfolder = (string)References[name]["subfolder"];
                // The slow classes the reference names, which AutoTokenizer would not pick here.
                Vocab Vocab() => SyncTask.Run(files.ReadJsonAsync<Vocab>(subfolder + "vocab.json"));
                string Merges() => SyncTask.Run(files.ReadTextAsync(subfolder + "merges.txt"));
                TokenizerConfig Config() => TokenizerConfig.Deserialize(SyncTask.Run(files.ReadTextAsync(subfolder + "tokenizer_config.json")));
                tokenizer = (string)References[name]["class"] switch {
                    "GPT2Tokenizer" => new GPT2Tokenizer(Vocab(), Merges(), Config()),
                    "CLIPTokenizer" => new ClipTokenizer(Vocab(), Merges(), Config()),
                    _ => SyncTask.Run(AutoTokenizer.FromPretrainedAsync(files, subfolder)),
                };
                Tokenizers[name] = tokenizer;
            }
            return tokenizer;
        }

        private static string Label(string text) {
            string escaped = text.Replace("\\", "\\\\").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
            return escaped.Length > 40 ? escaped[..40] + "..." : escaped;
        }

        public static IEnumerable<TestCaseData> Cases() {
            foreach (string name in Names) {
                JArray cases = (JArray)References[name]["cases"];
                for (int i = 0; i < cases.Count; i++) {
                    yield return new TestCaseData(name, i).SetName($"{name} {i:D2} \"{Label((string)cases[i]["text"])}\"");
                }
            }
        }

        public static IEnumerable<TestCaseData> Chats() {
            foreach (string name in Names) {
                JArray chats = (JArray)References[name]["chats"];
                for (int i = 0; i < chats.Count; i++) {
                    yield return new TestCaseData(name, i).SetName($"{name} chat {i}");
                }
            }
        }

        /// <summary>Known differences, with the reason.</summary>
        private static readonly Dictionary<(string, string), string> KnownIssues = new() {
            [("phi3", new string('a', 64))] = SentencepieceTies,
            [("phi3", new string('!', 20))] = SentencepieceTies,
        };

        private const string SentencepieceTies = "Microsoft.ML.Tokenizers merges runs of one character in another order than sentencepiece.";

        [TestCaseSource(nameof(Cases))]
        public void Encode_MatchesTransformers(string name, int index) {
            JToken c = References[name]["cases"][index];
            if (KnownIssues.TryGetValue((name, (string)c["text"]), out string issue)) {
                Assert.Ignore(issue);
            }
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

        [TestCaseSource(nameof(Chats))]
        public void ApplyChatTemplate_MatchesTransformers(string name, int index) {
            JToken chat = References[name]["chats"][index];
            var messages = chat["messages"].Select(m => new ChatMessage((string)m["role"], (string)m["content"]));
            string text = Tokenizer(name).ApplyChatTemplate(messages, (bool)chat["add_generation_prompt"]);
            Assert.That(text, Is.EqualTo((string)chat["text"]));
        }
    }
}
