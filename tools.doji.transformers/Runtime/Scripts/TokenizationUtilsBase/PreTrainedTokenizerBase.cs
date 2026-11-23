using Doji.AI.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    public enum Padding {
        /// <summary>No padding.</summary>
        None,
        /// <summary>Pads to the longest sequence of the batch. A single sequence is not padded.</summary>
        Longest,
        /// <summary>Pads to <c>maxLength</c>, or to <see cref="PreTrainedTokenizerBase.ModelMaxLength"/> without it.</summary>
        MaxLength
    }

    public enum Truncation {
        /// <summary>No truncation.</summary>
        None,
        /// <summary>Truncates the longest sequence of a pair first, so that both keep about the same length.</summary>
        LongestFirst,
        /// <summary>Truncates only the second sequence of a pair.</summary>
        OnlySecond,
        /// <summary>Truncates only the first sequence of a pair, or the single sequence.</summary>
        OnlyFirst
    }

    public enum Side { Right, Left }

    /// <summary>
    /// The base of all tokenizers: encoding with special tokens, truncation and padding, decoding, the special tokens
    /// and the chat template. <see cref="TokenizersBackend"/> implements it with the tokenizers library.
    /// </summary>
    public abstract partial class PreTrainedTokenizerBase {

        public const string TOKENIZER_CONFIG_FILE = "tokenizer_config.json";

        protected TokenizerConfig Config { get; }

        /// <summary>The longest input the model takes, the default length of truncation and padding.</summary>
        public int ModelMaxLength => Config.ModelMaxLength ?? int.MaxValue;
        public Side PaddingSide { get; set; }
        public Side TruncationSide { get; set; }
        /// <summary>The inputs the model takes, which <see cref="Encode"/> returns by default.</summary>
        public List<string> ModelInputNames { get; }
        public bool CleanUpTokenizationSpaces => Config.CleanUpTokenizationSpaces ?? false;

        /// <param name="defaultModelInputNames">The model inputs of the tokenizer class, which the config may change.</param>
        /// <param name="defaultPaddingSide">The padding side of the tokenizer class, which the config may change.</param>
        protected PreTrainedTokenizerBase(TokenizerConfig config, List<string> defaultModelInputNames = null, Side defaultPaddingSide = Side.Right) {
            Config = config ?? new TokenizerConfig();
            PaddingSide = Config.PaddingSide ?? defaultPaddingSide;
            TruncationSide = Config.TruncationSide ?? Side.Right;
            ModelInputNames = Config.ModelInputNames ?? defaultModelInputNames ?? new List<string> { "input_ids", "attention_mask" };
        }

        /// <summary>The number of tokens in the vocabulary, without the added tokens.</summary>
        public abstract int VocabSize { get; }

        /// <summary>The number of tokens in the vocabulary, with the added tokens.</summary>
        public abstract int Count { get; }

        /// <summary>The vocabulary with the added tokens, by token.</summary>
        public abstract Dictionary<string, int> GetVocab();

        /// <summary>Splits a text into tokens, without special tokens.</summary>
        public abstract List<string> Tokenize(string text);

        /// <summary>The id of a token, or the id of the unknown token if the vocabulary does not have it.</summary>
        public abstract int? ConvertTokensToIds(string token);

        public List<int?> ConvertTokensToIds(IEnumerable<string> tokens) => tokens.Select(ConvertTokensToIds).ToList();

        /// <summary>The token of an id, or null.</summary>
        public abstract string ConvertIdsToTokens(int id);

        public List<string> ConvertIdsToTokens(IEnumerable<int> ids, bool skipSpecialTokens = false) {
            var special = skipSpecialTokens ? new HashSet<int>(AllSpecialIds) : null;
            return ids.Where(id => special == null || !special.Contains(id)).Select(ConvertIdsToTokens).ToList();
        }

        /// <summary>Joins tokens into a text, as the decoder of the tokenizer does.</summary>
        public abstract string ConvertTokensToString(List<string> tokens);

        /// <summary>The ids of a text, without special tokens.</summary>
        protected abstract List<int> EncodeWithoutSpecialTokens(string text);

        /// <summary>The ids of a text split into words, without special tokens.</summary>
        protected abstract List<int> EncodeWithoutSpecialTokens(IEnumerable<string> words);

        /// <summary>The number of special tokens <see cref="Encode"/> adds to a sequence or a pair.</summary>
        public abstract int NumSpecialTokensToAdd(bool pair = false);

        /// <summary>
        /// Adds the special tokens to the ids of a sequence or a pair, and gives the type ids and the special tokens
        /// mask of the result.
        /// </summary>
        protected abstract (List<int> Ids, List<int> TypeIds, List<int> SpecialTokensMask) PostProcess(List<int> ids, List<int> pairIds, bool addSpecialTokens);

        /// <summary>
        /// Encodes a text or a batch of texts, and a pair, for the model.
        /// </summary>
        /// <param name="text">A text, a list of texts, a text split into words, or a list of those.</param>
        /// <param name="textPair">The second text of a pair, or a list of them for a batch.</param>
        /// <param name="addSpecialTokens">Whether to add the special tokens of the model, such as a BOS token.</param>
        /// <param name="maxLength">The length of truncation and of <see cref="Padding.MaxLength"/>. Without it, <see cref="ModelMaxLength"/>.</param>
        /// <param name="returnTokenTypeIds">Whether to return <c>token_type_ids</c>. By default, if <see cref="ModelInputNames"/> has them.</param>
        /// <param name="returnAttentionMask">Whether to return <c>attention_mask</c>. By default, if <see cref="ModelInputNames"/> has it.</param>
        public Encoding Encode(
            Input text,
            Input textPair = null,
            bool addSpecialTokens = true,
            Padding padding = Padding.None,
            Truncation truncation = Truncation.None,
            int? maxLength = null,
            int? padToMultipleOf = null,
            bool? returnTokenTypeIds = null,
            bool? returnAttentionMask = null,
            bool returnSpecialTokensMask = false,
            bool returnLength = false)
        {
            if (text == null) {
                throw new ArgumentNullException(nameof(text));
            }
            if (maxLength == null && (truncation != Truncation.None || padding == Padding.MaxLength) && ModelMaxLength < int.MaxValue) {
                maxLength = ModelMaxLength;
            }
            if (padding != Padding.None && PadTokenId == null) {
                throw new InvalidOperationException("Padding needs a pad token, and the tokenizer has none.");
            }
            if (padding == Padding.MaxLength && maxLength == null) {
                throw new ArgumentException("Padding to the max length needs a max length, and the model has none.");
            }
            var options = new EncodeOptions {
                AddSpecialTokens = addSpecialTokens,
                Truncation = truncation,
                MaxLength = maxLength,
                ReturnTokenTypeIds = returnTokenTypeIds ?? ModelInputNames.Contains("token_type_ids"),
                ReturnAttentionMask = returnAttentionMask ?? ModelInputNames.Contains("attention_mask"),
                ReturnSpecialTokensMask = returnSpecialTokensMask,
            };

            if (!text.IsBatch()) {
                InputEncoding single = EncodeOne(Ids(text), textPair != null ? Ids(textPair) : null, options);
                if (padding == Padding.MaxLength) {
                    Pad(single, maxLength.Value, padToMultipleOf, options.ReturnAttentionMask);
                } else if (padToMultipleOf != null && padding == Padding.Longest) {
                    Pad(single, single.InputIds.Count(), padToMultipleOf, options.ReturnAttentionMask);
                }
                if (returnLength) {
                    single["length"] = single.InputIds.Count();
                }
                return single;
            }

            var texts = ((IBatchInput)text).Sequence.Cast<object>().ToList();
            List<object> pairs = null;
            if (textPair != null) {
                if (!textPair.IsBatch() || ((IBatchInput)textPair).Sequence.Count != texts.Count) {
                    throw new ArgumentException("When encoding a batch, the pairs must be a list as long as the texts.");
                }
                pairs = ((IBatchInput)textPair).Sequence.Cast<object>().ToList();
            }
            var encodings = new List<InputEncoding>();
            for (int i = 0; i < texts.Count; i++) {
                encodings.Add(EncodeOne(Ids(texts[i]), pairs != null ? Ids(pairs[i]) : null, options));
            }
            int? padTo = padding switch {
                Padding.MaxLength => maxLength,
                Padding.Longest => encodings.Max(e => e.InputIds.Count()),
                _ => null,
            };
            var batch = new BatchEncoding();
            foreach (InputEncoding encoding in encodings) {
                if (padTo != null) {
                    Pad(encoding, padTo.Value, padToMultipleOf, options.ReturnAttentionMask);
                }
                if (returnLength) {
                    encoding["length"] = new List<int> { encoding.InputIds.Count() };
                }
                batch.Append(encoding);
            }
            return batch;
        }

        private List<int> Ids(object input) {
            return input switch {
                string s => EncodeWithoutSpecialTokens(s),
                SingleInput s => EncodeWithoutSpecialTokens(s.Text),
                List<string> words => EncodeWithoutSpecialTokens(words),
                PretokenizedSingleInput p => EncodeWithoutSpecialTokens(p.PretokenizedText),
                _ => throw new ArgumentException($"Cannot encode a {input?.GetType().Name ?? "null"}."),
            };
        }

        private struct EncodeOptions {
            public bool AddSpecialTokens;
            public Truncation Truncation;
            public int? MaxLength;
            public bool ReturnTokenTypeIds;
            public bool ReturnAttentionMask;
            public bool ReturnSpecialTokensMask;
        }

        /// <summary>
        /// Truncates the ids, keeping room for the special tokens, then adds the special tokens, as the tokenizers
        /// library does.
        /// </summary>
        private InputEncoding EncodeOne(List<int> ids, List<int> pairIds, EncodeOptions options) {
            if (options.Truncation != Truncation.None && options.MaxLength != null) {
                int numSpecial = options.AddSpecialTokens ? NumSpecialTokensToAdd(pairIds != null) : 0;
                int maxLength = options.MaxLength.Value - numSpecial;
                if (maxLength < 0) {
                    throw new ArgumentException($"The max length {options.MaxLength} is shorter than the {numSpecial} special tokens.");
                }
                Truncate(ref ids, ref pairIds, maxLength, options.Truncation);
            }
            var (sequence, typeIds, specialMask) = PostProcess(ids, pairIds, options.AddSpecialTokens);
            var encoding = new InputEncoding { ["input_ids"] = sequence };
            if (options.ReturnTokenTypeIds) {
                encoding["token_type_ids"] = typeIds;
            }
            if (options.ReturnSpecialTokensMask) {
                encoding["special_tokens_mask"] = specialMask;
            }
            if (options.ReturnAttentionMask) {
                encoding["attention_mask"] = Enumerable.Repeat(1, sequence.Count).ToList();
            }
            return encoding;
        }

        /// <summary>The truncation of the tokenizers library (<c>truncate_encodings</c>).</summary>
        private void Truncate(ref List<int> ids, ref List<int> pairIds, int maxLength, Truncation strategy) {
            int total = ids.Count + (pairIds?.Count ?? 0);
            if (total <= maxLength) {
                return;
            }
            if (pairIds == null) {
                if (strategy == Truncation.OnlySecond) {
                    throw new ArgumentException("Cannot truncate only the second sequence of a single sequence.");
                }
                ids = TruncateTo(ids, maxLength);
                return;
            }
            switch (strategy) {
                case Truncation.LongestFirst: {
                    int n1 = ids.Count, n2 = pairIds.Count;
                    bool swap = n1 > n2;
                    if (swap) {
                        (n1, n2) = (n2, n1);
                    }
                    n2 = n1 > maxLength ? n1 : Math.Max(n1, maxLength - n1);
                    if (n1 + n2 > maxLength) {
                        n1 = maxLength / 2;
                        n2 = n1 + maxLength % 2;
                    }
                    if (swap) {
                        (n1, n2) = (n2, n1);
                    }
                    ids = TruncateTo(ids, n1);
                    pairIds = TruncateTo(pairIds, n2);
                    break;
                }
                case Truncation.OnlyFirst:
                case Truncation.OnlySecond: {
                    bool first = strategy == Truncation.OnlyFirst;
                    int other = first ? pairIds.Count : ids.Count;
                    if (other >= maxLength) {
                        throw new ArgumentException("The sequence to truncate is too short to respect the max length.");
                    }
                    if (first) {
                        ids = TruncateTo(ids, maxLength - other);
                    } else {
                        pairIds = TruncateTo(pairIds, maxLength - other);
                    }
                    break;
                }
            }
        }

        private List<int> TruncateTo(List<int> ids, int length) {
            if (ids.Count <= length) {
                return ids;
            }
            return TruncationSide == Side.Right ? ids.GetRange(0, length) : ids.GetRange(ids.Count - length, length);
        }

        /// <summary>Pads one encoding to <paramref name="length"/>, rounded up to a multiple of <paramref name="padToMultipleOf"/>.</summary>
        private void Pad(InputEncoding encoding, int length, int? padToMultipleOf, bool attentionMask) {
            if (padToMultipleOf is int multiple && multiple > 0 && length % multiple != 0) {
                length = (length / multiple + 1) * multiple;
            }
            var ids = (List<int>)encoding["input_ids"];
            int difference = length - ids.Count;
            if (difference <= 0) {
                return;
            }
            int padId = PadTokenId ?? throw new InvalidOperationException("Padding needs a pad token, and the tokenizer has none.");
            void PadList(string key, int value) {
                if (encoding.TryGetValue(key, out object list)) {
                    var values = (List<int>)list;
                    values.InsertRange(PaddingSide == Side.Right ? values.Count : 0, Enumerable.Repeat(value, difference));
                }
            }
            if (attentionMask && !encoding.ContainsKey("attention_mask")) {
                encoding["attention_mask"] = Enumerable.Repeat(1, ids.Count).ToList();
            }
            PadList("attention_mask", 0);
            PadList("token_type_ids", PadTokenTypeId);
            PadList("special_tokens_mask", 1);
            PadList("input_ids", padId);
        }

        /// <summary>Decodes ids to text.</summary>
        /// <param name="skipSpecialTokens">Whether to leave out the special tokens.</param>
        /// <param name="cleanUpTokenizationSpaces">Whether to remove the spaces before punctuation. By default, <see cref="CleanUpTokenizationSpaces"/>.</param>
        public abstract string Decode(IEnumerable<int> tokenIds, bool skipSpecialTokens = false, bool? cleanUpTokenizationSpaces = null);

        public List<string> BatchDecode(IEnumerable<IEnumerable<int>> sequences, bool skipSpecialTokens = false, bool? cleanUpTokenizationSpaces = null) {
            return sequences.Select(ids => Decode(ids, skipSpecialTokens, cleanUpTokenizationSpaces)).ToList();
        }

        /// <summary>
        /// Removes the spaces before punctuation and in English contractions, which word-level tokenizers leave.
        /// </summary>
        public static string CleanUpTokenization(string text) {
            return text
                .Replace(" .", ".")
                .Replace(" ?", "?")
                .Replace(" !", "!")
                .Replace(" ,", ",")
                .Replace(" ' ", "'")
                .Replace(" n't", "n't")
                .Replace(" 'm", "'m")
                .Replace(" 's", "'s")
                .Replace(" 've", "'ve")
                .Replace(" 're", "'re");
        }
    }
}
