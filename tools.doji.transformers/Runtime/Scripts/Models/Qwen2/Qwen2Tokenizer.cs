using Doji.AI.Tokenizers;
using Doji.AI.Tokenizers.Models;
using Doji.AI.Tokenizers.PostProcessors;
using System.Collections.Generic;
using System.Linq;
using Decoders = Doji.AI.Tokenizers.Decoders;
using Normalizers = Doji.AI.Tokenizers.Normalizers;
using PreTokenizers = Doji.AI.Tokenizers.PreTokenizers;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The byte-level BPE tokenizer of Qwen2, Qwen2.5 and Qwen3, as transformers' <c>Qwen2Tokenizer</c> builds it.
    /// </summary>
    public class Qwen2Tokenizer : TokenizersBackend {

        private const string Pattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

        /// <param name="vocab">The vocabulary, as in <c>vocab.json</c>.</param>
        /// <param name="merges">The merges by rank, as in <c>merges.txt</c>.</param>
        /// <param name="postProcessor">The post-processor of <c>tokenizer.json</c>, if the vocabulary comes from it.</param>
        public Qwen2Tokenizer(Dictionary<string, int> vocab, IEnumerable<(string, string)> merges, TokenizerConfig config = null,
            PostProcessor postProcessor = null) : base(config, new List<string> { "input_ids", "attention_mask" }) {
            SetSpecialTokens(unk: "<|endoftext|>", eos: "<|endoftext|>", pad: "<|endoftext|>");
            var backend = new Tokenizer(new BPE(vocab, merges, continuingSubwordPrefix: "", endOfWordSuffix: "")) {
                Normalizer = new Normalizers.NFC(),
                PreTokenizer = new PreTokenizers.Sequence(new PreTokenizers.PreTokenizer[] {
                    new PreTokenizers.Split(Normalizers.Pattern.FromRegex(Pattern), PreTokenizers.SplitDelimiterBehavior.Isolated),
                    new PreTokenizers.ByteLevel(addPrefixSpace: Config.AddPrefixSpace ?? false, useRegex: false),
                }),
                Decoder = new Decoders.ByteLevel(),
            };
            Initialize(backend, postProcessor);
            AddTokens(AllSpecialTokens.Select(t => new AddedToken(t, special: true)));
        }

        /// <param name="vocabJson">The content of <c>vocab.json</c>.</param>
        /// <param name="mergesTxt">The content of <c>merges.txt</c>.</param>
        public Qwen2Tokenizer(string vocabJson, string mergesTxt, TokenizerConfig config = null)
            : this(Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabJson), ReadMerges(mergesTxt), config) { }

        /// <summary>Takes the vocabulary and the post-processor of <c>tokenizer.json</c>.</summary>
        public static Qwen2Tokenizer FromTokenizerJson(string tokenizerJson, TokenizerConfig config = null) {
            var (vocab, merges, postProcessor) = ReadBpe(tokenizerJson);
            return new Qwen2Tokenizer(vocab, merges, config, postProcessor);
        }
    }
}
