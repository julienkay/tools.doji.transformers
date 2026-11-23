using Doji.AI.Tokenizers;
using Doji.AI.Tokenizers.Models;
using Doji.AI.Tokenizers.PostProcessors;
using System.Collections.Generic;
using Decoders = Doji.AI.Tokenizers.Decoders;
using Normalizers = Doji.AI.Tokenizers.Normalizers;
using PreTokenizers = Doji.AI.Tokenizers.PreTokenizers;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The tokenizer of CLIP, as transformers' <c>CLIPTokenizer</c> builds it: lowercase byte-level BPE with an
    /// end-of-word suffix, between a start and an end token.
    /// </summary>
    public class CLIPTokenizer : TokenizersBackend {

        private const string Pattern = @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";
        private const string EndOfWordSuffix = "</w>";

        /// <param name="vocab">The vocabulary, as in <c>vocab.json</c>.</param>
        /// <param name="merges">The merges by rank, as in <c>merges.txt</c>.</param>
        public CLIPTokenizer(Dictionary<string, int> vocab, IEnumerable<(string, string)> merges, TokenizerConfig config = null)
            : base(config, new List<string> { "input_ids", "attention_mask" }) {
            SetSpecialTokens(unk: "<|endoftext|>", bos: "<|startoftext|>", eos: "<|endoftext|>", pad: "<|endoftext|>");
            var backend = new Tokenizer(new BPE(vocab, merges, unkToken: UnkToken, continuingSubwordPrefix: "", endOfWordSuffix: EndOfWordSuffix)) {
                Normalizer = new Normalizers.Sequence(new Normalizers.Normalizer[] {
                    new Normalizers.NFC(),
                    new Normalizers.Replace(Normalizers.Pattern.FromRegex(@"\s+"), " "),
                    new Normalizers.Lowercase(),
                }),
                PreTokenizer = new PreTokenizers.Sequence(new PreTokenizers.PreTokenizer[] {
                    new PreTokenizers.Split(Normalizers.Pattern.FromRegex(Pattern), PreTokenizers.SplitDelimiterBehavior.Removed, invert: true),
                    new PreTokenizers.ByteLevel(addPrefixSpace: false),
                }),
                Decoder = new Decoders.ByteLevel(),
            };
            Initialize(backend, null);
            backend.PostProcessor = new RobertaProcessing((EosToken, EosTokenId.Value), (BosToken, BosTokenId.Value));
        }

        /// <param name="vocabJson">The content of <c>vocab.json</c>.</param>
        /// <param name="mergesTxt">The content of <c>merges.txt</c>.</param>
        public CLIPTokenizer(string vocabJson, string mergesTxt, TokenizerConfig config = null)
            : this(Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabJson), ReadMerges(mergesTxt), config) { }

        /// <summary>Takes the vocabulary of <c>tokenizer.json</c>.</summary>
        public static CLIPTokenizer FromTokenizerJson(string tokenizerJson, TokenizerConfig config = null) {
            var (vocab, merges, _) = ReadBpe(tokenizerJson);
            return new CLIPTokenizer(vocab, merges, config);
        }

        /// <summary>The end-of-word suffixes become spaces, as transformers makes the backend decode.</summary>
        protected override string DecodeBackend(IEnumerable<int> tokenIds, bool skipSpecialTokens) {
            return base.DecodeBackend(tokenIds, skipSpecialTokens).Replace(EndOfWordSuffix, " ").Trim();
        }
    }
}
