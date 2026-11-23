using Doji.AI.Tokenizers;
using Doji.AI.Tokenizers.Models;
using Doji.AI.Tokenizers.PostProcessors;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A tokenizer on top of a <see cref="Tokenizer"/> of the tokenizers library, as all tokenizers of transformers v5
    /// are. Loaded from <c>tokenizer.json</c>, it keeps the whole pipeline of the file. The model classes, such as
    /// <see cref="LlamaTokenizer"/>, build the pipeline of their model instead and only take the vocabulary from the file.
    /// </summary>
    public class TokenizersBackend : PreTrainedTokenizerBase {

        /// <summary>The tokenizer of the tokenizers library that does the work.</summary>
        public Tokenizer Backend { get; private set; }

        private bool? _addBosToken;
        private bool? _addEosToken;

        /// <param name="tokenizerJson">The content of <c>tokenizer.json</c>.</param>
        /// <param name="config">The content of <c>tokenizer_config.json</c>.</param>
        public TokenizersBackend(string tokenizerJson, TokenizerConfig config = null)
            : this(Tokenizer.FromJson(tokenizerJson), config) { }

        public TokenizersBackend(Tokenizer backend, TokenizerConfig config = null) : base(config) {
            SetSpecialTokens();
            Initialize(backend, backend.PostProcessor);
        }

        /// <summary>For model classes, which call <see cref="Initialize"/> once they have built their backend.</summary>
        protected TokenizersBackend(TokenizerConfig config, List<string> modelInputNames, Side paddingSide = Side.Right)
            : base(config, modelInputNames, paddingSide) { }

        /// <summary>
        /// Adds the added tokens and the special tokens of the config to the backend, and builds its post-processor from
        /// <c>add_bos_token</c> and <c>add_eos_token</c> when the config sets them or there is none.
        /// </summary>
        /// <param name="postProcessor">The post-processor of <c>tokenizer.json</c>, if there is one.</param>
        protected void Initialize(Tokenizer backend, PostProcessor postProcessor) {
            Backend = backend;
            Backend.PostProcessor = postProcessor;
            Backend.EncodeSpecialTokens = Config.SplitSpecialTokens ?? false;
            _addBosToken = Config.AddBosToken;
            _addEosToken = Config.AddEosToken;

            var present = Backend.AddedTokensDecoder.Values.ToList();
            var tokensToAdd = new List<AddedToken>();
            if (Config.AddedTokensDecoder != null) {
                foreach (var entry in Config.AddedTokensDecoder.OrderBy(e => e.Key)) {
                    if (!present.Any(t => Same(t, entry.Value))) {
                        tokensToAdd.Add(entry.Value);
                    }
                }
            }
            var encoder = new HashSet<string>(present.Select(t => t.Content).Concat(tokensToAdd.Select(t => t.Content)));
            var named = new HashSet<string>(SpecialTokensMap.Values.Where(t => t != null).Select(t => t.Content));
            foreach (AddedToken token in SpecialTokensMap.Values.Concat(ExtraSpecialTokens)) {
                if (token != null && !encoder.Contains(token.Content) && !tokensToAdd.Any(t => Same(t, token))) {
                    if (!token.Special && named.Contains(token.Content)) {
                        token.Special = true;
                    }
                    tokensToAdd.Add(token);
                }
            }
            if (tokensToAdd.Count > 0) {
                Backend.AddTokens(tokensToAdd);
            }
            if (_addBosToken.HasValue || _addEosToken.HasValue || Backend.PostProcessor == null) {
                UpdatePostProcessor();
            }
        }

        private static bool Same(AddedToken a, AddedToken b) {
            return a.Content == b.Content && a.Special == b.Special && a.Normalized == b.Normalized
                && a.SingleWord == b.SingleWord && a.LStrip == b.LStrip && a.RStrip == b.RStrip;
        }

        /// <summary>Whether <see cref="PreTrainedTokenizerBase.Encode"/> adds the BOS token.</summary>
        public bool AddBosToken {
            get => _addBosToken ?? false;
            set {
                _addBosToken = value;
                UpdatePostProcessor();
            }
        }

        /// <summary>Whether <see cref="PreTrainedTokenizerBase.Encode"/> adds the EOS token.</summary>
        public bool AddEosToken {
            get => _addEosToken ?? false;
            set {
                _addEosToken = value;
                UpdatePostProcessor();
            }
        }

        /// <summary>Builds a template post-processor that adds the BOS and EOS tokens as set.</summary>
        protected void UpdatePostProcessor() {
            string bos = BosToken, eos = EosToken;
            bool addBos = AddBosToken && bos != null;
            bool addEos = AddEosToken && eos != null;
            string single = $"{(addBos ? bos + ":0 " : "")}$A:0{(addEos ? " " + eos + ":0" : "")}";
            string pair = $"{single}{(addBos ? " " + bos + ":1" : "")} $B:1{(addEos ? " " + eos + ":1" : "")}";
            var specialTokens = new List<(string, int)>();
            if (addBos) {
                specialTokens.Add((bos, BosTokenId ?? throw new InvalidOperationException($"The BOS token '{bos}' is not in the vocabulary.")));
            }
            if (addEos && eos != bos) {
                specialTokens.Add((eos, EosTokenId ?? throw new InvalidOperationException($"The EOS token '{eos}' is not in the vocabulary.")));
            }
            Backend.PostProcessor = TemplateProcessing.Create(single, pair, specialTokens);
        }

        public override int VocabSize => Backend.GetVocabSize(false);
        public override int Count => Backend.GetVocabSize(true);
        public override Dictionary<string, int> GetVocab() => Backend.GetVocab();

        /// <summary>The added tokens by id.</summary>
        public IReadOnlyDictionary<int, AddedToken> AddedTokensDecoder => Backend.AddedTokensDecoder;

        /// <summary>Adds tokens to the vocabulary. Returns how many it added.</summary>
        /// <param name="specialTokens">Whether the tokens are special, which decoding can skip.</param>
        public int AddTokens(IEnumerable<AddedToken> tokens, bool specialTokens = false) {
            return specialTokens ? Backend.AddSpecialTokens(tokens) : Backend.AddTokens(tokens);
        }

        public int AddTokens(params string[] tokens) => AddTokens(tokens.Select(t => new AddedToken(t)));

        public override List<string> Tokenize(string text) => Backend.Encode(text, addSpecialTokens: false).Tokens;

        public override int? ConvertTokensToIds(string token) {
            return token == null ? null : Backend.TokenToId(token) ?? (UnkToken != null ? Backend.TokenToId(UnkToken) : null);
        }

        public override string ConvertIdsToTokens(int id) => Backend.IdToToken(id);

        public override string ConvertTokensToString(List<string> tokens) {
            return Backend.Decoder != null ? Backend.Decoder.Decode(tokens) : string.Join(" ", tokens);
        }

        protected override List<int> EncodeWithoutSpecialTokens(string text) => Backend.Encode(text, addSpecialTokens: false).Ids;

        protected override List<int> EncodeWithoutSpecialTokens(IEnumerable<string> words) => Backend.Encode(words, addSpecialTokens: false).Ids;

        public override int NumSpecialTokensToAdd(bool pair = false) => Backend.NumSpecialTokensToAdd(pair);

        protected override (List<int> Ids, List<int> TypeIds, List<int> SpecialTokensMask) PostProcess(List<int> ids, List<int> pairIds, bool addSpecialTokens) {
            var encoding = Backend.PostProcess(ids, pairIds, addSpecialTokens);
            return (encoding.Ids, encoding.TypeIds, encoding.SpecialTokensMask);
        }

        public override string Decode(IEnumerable<int> tokenIds, bool skipSpecialTokens = false, bool? cleanUpTokenizationSpaces = null) {
            string text = DecodeBackend(tokenIds, skipSpecialTokens);
            // The clean up is for word-level tokenizers, and breaks the text of BPE ones.
            if ((cleanUpTokenizationSpaces ?? CleanUpTokenizationSpaces) && Backend.Model is not BPE) {
                text = CleanUpTokenization(text);
            }
            return text;
        }

        /// <summary>The decoding of the backend, which a model class may change.</summary>
        protected virtual string DecodeBackend(IEnumerable<int> tokenIds, bool skipSpecialTokens) {
            return Backend.Decode(tokenIds, skipSpecialTokens);
        }

        /// <summary>
        /// The parts of <c>tokenizer.json</c> that the model classes take: the vocabulary and merges of its BPE model,
        /// and its post-processor.
        /// </summary>
        internal static (Dictionary<string, int> Vocab, List<(string, string)> Merges, PostProcessor PostProcessor) ReadBpe(string tokenizerJson) {
            JObject json = JObject.Parse(tokenizerJson);
            JObject model = (JObject)json["model"];
            if (((string)model["type"] ?? "BPE") != "BPE") {
                throw new NotSupportedException($"Expected a BPE model, got '{model["type"]}'.");
            }
            var vocab = model["vocab"].ToObject<Dictionary<string, int>>();
            var merges = new List<(string, string)>();
            foreach (JToken merge in (JArray)model["merges"] ?? new JArray()) {
                if (merge is JArray parts) {
                    merges.Add(((string)parts[0], (string)parts[1]));
                } else {
                    string text = (string)merge;
                    int space = text.IndexOf(' ');
                    merges.Add((text[..space], text[(space + 1)..]));
                }
            }
            return (vocab, merges, PostProcessor.FromJson(json["post_processor"]));
        }

        /// <summary>Reads <c>merges.txt</c>: one merge per line, after an optional <c>#version</c> line.</summary>
        internal static List<(string, string)> ReadMerges(string mergesTxt) {
            var merges = new List<(string, string)>();
            foreach (string line in mergesTxt.Split('\n')) {
                string merge = line.TrimEnd('\r');
                if (merge.Length == 0 || merge.StartsWith("#version", StringComparison.Ordinal)) {
                    continue;
                }
                int space = merge.IndexOf(' ');
                merges.Add((merge[..space], merge[(space + 1)..]));
            }
            return merges;
        }
    }
}
