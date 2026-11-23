using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A tokenizer read from the <c>tokenizer.json</c> of the Hugging Face tokenizers library, as transformers'
    /// fast tokenizers are. Supports byte-level BPE models, as GPT-2, SmolLM2, Qwen2 and Llama 3 use: it splits out
    /// the added tokens, normalizes, pre-tokenizes, maps the bytes to unicode and merges by rank.
    /// </summary>
    public class PreTrainedTokenizerFast : PreTrainedTokenizerBase {

        public override bool Fast => true;

        private readonly Dictionary<string, int> _vocab;
        private readonly string[] _idToToken;
        private readonly Dictionary<(string, string), int> _mergeRanks;
        private readonly bool _ignoreMerges;
        private readonly Dictionary<string, int> _addedTokens = new();
        private readonly HashSet<int> _specialIds = new();
        private readonly Regex _addedTokenPattern;
        private readonly bool _nfc;
        private readonly List<Func<string, IEnumerable<string>>> _preTokenizers = new();
        private readonly bool _addPrefixSpace;
        private readonly Dictionary<string, string[]> _cache = new();

        private static readonly char[] ByteToChar = BytesToUnicode();
        private static readonly Dictionary<char, byte> CharToByte = Enumerable.Range(0, 256).ToDictionary(b => ByteToChar[b], b => (byte)b);

        /// <summary>
        /// The pre-tokenizer regex of GPT-2, which ByteLevel uses when <c>use_regex</c> is true.
        /// </summary>
        private const string Gpt2Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

        /// <param name="tokenizerJson">The content of <c>tokenizer.json</c>.</param>
        /// <param name="config">The content of <c>tokenizer_config.json</c>, for the special tokens and the chat template.</param>
        public PreTrainedTokenizerFast(string tokenizerJson, TokenizerConfig config) : base(config) {
            JObject tokenizer = JObject.Parse(tokenizerJson);
            JObject model = (JObject)tokenizer["model"];
            if ((string)model["type"] != "BPE") {
                throw new NotSupportedException($"Expected a BPE tokenizer, got '{model["type"]}'.");
            }
            if ((bool?)model["byte_fallback"] ?? false) {
                throw new NotSupportedException("BPE with byte fallback (sentencepiece-style) is not supported. Use LlamaTokenizer.");
            }
            _vocab = model["vocab"].ToObject<Dictionary<string, int>>();
            _ignoreMerges = (bool?)model["ignore_merges"] ?? false;

            // tokenizers writes a merge as "a b" or, since 0.20, as ["a", "b"].
            JArray merges = (JArray)model["merges"];
            _mergeRanks = new Dictionary<(string, string), int>(merges.Count);
            for (int i = 0; i < merges.Count; i++) {
                (string, string) pair;
                if (merges[i] is JArray parts) {
                    pair = ((string)parts[0], (string)parts[1]);
                } else {
                    string merge = (string)merges[i];
                    int space = merge.IndexOf(' ');
                    pair = (merge[..space], merge[(space + 1)..]);
                }
                _mergeRanks.TryAdd(pair, i);
            }

            foreach (JToken token in tokenizer["added_tokens"] ?? new JArray()) {
                string content = (string)token["content"];
                int id = (int)token["id"];
                _addedTokens[content] = id;
                _vocab[content] = id;
                if ((bool?)token["special"] ?? false) {
                    _specialIds.Add(id);
                }
            }
            _idToToken = new string[_vocab.Values.Max() + 1];
            foreach (var entry in _vocab) {
                _idToToken[entry.Value] = entry.Key;
            }
            if (_addedTokens.Count > 0) {
                string alternatives = string.Join("|", _addedTokens.Keys.OrderByDescending(t => t.Length).Select(Regex.Escape));
                _addedTokenPattern = new Regex($"({alternatives})", RegexOptions.Compiled);
            }

            _nfc = ReadNormalizer(tokenizer["normalizer"]);
            AddPreTokenizer(tokenizer["pre_tokenizer"], ref _addPrefixSpace);
            if (tokenizer["post_processor"] is JObject post && (string)post["type"] != "ByteLevel") {
                throw new NotSupportedException($"The post-processor '{post["type"]}' is not supported.");
            }
            Initialize();
        }

        private static bool ReadNormalizer(JToken normalizer) {
            if (normalizer == null || normalizer.Type == JTokenType.Null) {
                return false;
            }
            return (string)normalizer["type"] switch {
                "NFC" => true,
                _ => throw new NotSupportedException($"The normalizer '{normalizer["type"]}' is not supported."),
            };
        }

        private void AddPreTokenizer(JToken pre, ref bool addPrefixSpace) {
            if (pre == null || pre.Type == JTokenType.Null) {
                return;
            }
            switch ((string)pre["type"]) {
                case "Sequence":
                    foreach (JToken child in pre["pretokenizers"]) {
                        AddPreTokenizer(child, ref addPrefixSpace);
                    }
                    break;
                case "ByteLevel":
                    addPrefixSpace |= (bool?)pre["add_prefix_space"] ?? false;
                    if ((bool?)pre["use_regex"] ?? true) {
                        var gpt2 = new Regex(Gpt2Pattern, RegexOptions.Compiled);
                        _preTokenizers.Add(text => gpt2.Matches(text).Select(m => m.Value));
                    }
                    break;
                case "Digits":
                    var digits = new Regex((bool?)pre["individual_digits"] ?? false ? @"\p{N}" : @"\p{N}+", RegexOptions.Compiled);
                    _preTokenizers.Add(text => Isolate(digits, text));
                    break;
                case "Split":
                    if ((string)pre["behavior"] != "Isolated" || ((bool?)pre["invert"] ?? false)) {
                        throw new NotSupportedException($"The Split pre-tokenizer behavior '{pre["behavior"]}' is not supported.");
                    }
                    string pattern = (string)pre["pattern"]["Regex"] ?? Regex.Escape((string)pre["pattern"]["String"]);
                    var split = new Regex(pattern, RegexOptions.Compiled);
                    _preTokenizers.Add(text => Isolate(split, text));
                    break;
                default:
                    throw new NotSupportedException($"The pre-tokenizer '{pre["type"]}' is not supported.");
            }
        }

        /// <summary>
        /// Splits <paramref name="text"/> so that each match and each gap between matches is one piece.
        /// </summary>
        private static IEnumerable<string> Isolate(Regex regex, string text) {
            int start = 0;
            foreach (Match match in regex.Matches(text)) {
                if (match.Index > start) {
                    yield return text[start..match.Index];
                }
                if (match.Length > 0) {
                    yield return match.Value;
                }
                start = match.Index + match.Length;
            }
            if (start < text.Length) {
                yield return text[start..];
            }
        }

        protected override Dictionary<string, int> GetVocab() => _vocab;

        protected override int NumSpecialTokensToAdd(bool pair = false) => 0;

        protected override Encoding EncodePlus(EncodingParams args) {
            if (args.Text is not SingleInput single || args.TextPair != null) {
                throw new NotSupportedException("This tokenizer encodes one text at a time.");
            }
            return PrepareForModel(args, ConvertTokensToIds(Tokenize(single.Text, args)));
        }

        /// <summary>
        /// Splits <paramref name="text"/> into tokens. Added tokens such as <c>&lt;|im_start|&gt;</c> in the text stay whole.
        /// </summary>
        protected override List<string> Tokenize(string text, EncodingParams args) {
            var tokens = new List<string>();
            IEnumerable<string> segments = _addedTokenPattern != null ? _addedTokenPattern.Split(text) : new[] { text };
            foreach (string segment in segments) {
                if (segment.Length == 0) {
                    continue;
                }
                if (_addedTokens.ContainsKey(segment)) {
                    tokens.Add(segment);
                    continue;
                }
                string normalized = _nfc ? segment.Normalize(NormalizationForm.FormC) : segment;
                if (_addPrefixSpace && !normalized.StartsWith(" ")) {
                    normalized = " " + normalized;
                }
                IEnumerable<string> words = new[] { normalized };
                foreach (var preTokenizer in _preTokenizers) {
                    words = words.SelectMany(preTokenizer).ToList();
                }
                foreach (string word in words) {
                    tokens.AddRange(Bpe(word));
                }
            }
            return tokens;
        }

        protected override List<int> ConvertTokensToIds(List<string> tokens) => tokens.Select(ConvertTokensToIds).ToList();

        protected override int ConvertTokensToIds(string token) {
            return token != null && _vocab.TryGetValue(token, out int id) ? id : (UnkToken != null && _vocab.TryGetValue(UnkToken, out int unk) ? unk : -1);
        }

        public override string Decode(List<int> tokenIds, bool skipSpecialTokens = false, bool? cleanUpTokenizationSpaces = null, bool spacesBetweenSpecialTokens = true) {
            var text = new StringBuilder();
            var tokens = new List<string>();
            foreach (int id in tokenIds) {
                if (skipSpecialTokens && _specialIds.Contains(id)) {
                    continue;
                }
                string token = id >= 0 && id < _idToToken.Length ? _idToToken[id] : null;
                if (token == null) {
                    continue;
                }
                if (_addedTokens.ContainsKey(token)) {
                    text.Append(ConvertTokensToString(tokens)).Append(token);
                    tokens.Clear();
                } else {
                    tokens.Add(token);
                }
            }
            text.Append(ConvertTokensToString(tokens));
            return cleanUpTokenizationSpaces ?? CleanUpTokenizationSpaces ? CleanUpTokenization(text.ToString()) : text.ToString();
        }

        /// <summary>
        /// Maps the unicode characters of byte-level tokens back to bytes, and decodes them as UTF-8.
        /// </summary>
        protected override string ConvertTokensToString(List<string> tokens) {
            var bytes = new List<byte>();
            foreach (string token in tokens) {
                foreach (char c in token) {
                    bytes.Add(CharToByte[c]);
                }
            }
            return System.Text.Encoding.UTF8.GetString(bytes.ToArray());
        }

        /// <summary>
        /// The tokens of one pre-tokenized word: its UTF-8 bytes mapped to unicode, then merged by rank.
        /// </summary>
        private string[] Bpe(string word) {
            if (_cache.TryGetValue(word, out string[] cached)) {
                return cached;
            }
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(word);
            var symbols = new List<string>(bytes.Length);
            foreach (byte b in bytes) {
                symbols.Add(ByteToChar[b].ToString());
            }
            string mapped = string.Concat(symbols);
            if (_ignoreMerges && _vocab.ContainsKey(mapped)) {
                return _cache[word] = new[] { mapped };
            }

            while (symbols.Count > 1) {
                int best = -1, bestRank = int.MaxValue;
                for (int i = 0; i < symbols.Count - 1; i++) {
                    if (_mergeRanks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank) {
                        best = i;
                        bestRank = rank;
                    }
                }
                if (best < 0) {
                    break;
                }
                // Merge every occurrence of the best pair, left to right, as the reference BPE does.
                string first = symbols[best], second = symbols[best + 1];
                var merged = new List<string>(symbols.Count);
                for (int i = 0; i < symbols.Count; i++) {
                    if (i < symbols.Count - 1 && symbols[i] == first && symbols[i + 1] == second) {
                        merged.Add(first + second);
                        i++;
                    } else {
                        merged.Add(symbols[i]);
                    }
                }
                symbols = merged;
            }
            return _cache[word] = symbols.ToArray();
        }

        /// <summary>
        /// GPT-2's map from bytes to printable unicode characters, which keeps whitespace and control bytes out of the vocabulary.
        /// </summary>
        private static char[] BytesToUnicode() {
            var map = new char[256];
            int n = 0;
            for (int b = 0; b < 256; b++) {
                bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
                map[b] = printable ? (char)b : (char)(256 + n++);
            }
            return map;
        }
    }
}
