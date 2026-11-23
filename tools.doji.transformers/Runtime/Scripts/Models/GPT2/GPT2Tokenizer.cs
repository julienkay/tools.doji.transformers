using Doji.AI.Tokenizers;
using Doji.AI.Tokenizers.Models;
using Doji.AI.Tokenizers.PostProcessors;
using System.Collections.Generic;
using Decoders = Doji.AI.Tokenizers.Decoders;
using PreTokenizers = Doji.AI.Tokenizers.PreTokenizers;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The byte-level BPE tokenizer of GPT-2, as transformers' <c>GPT2Tokenizer</c> builds it.
    /// </summary>
    public class GPT2Tokenizer : TokenizersBackend {

        /// <param name="vocab">The vocabulary, as in <c>vocab.json</c>.</param>
        /// <param name="merges">The merges by rank, as in <c>merges.txt</c>.</param>
        /// <param name="postProcessor">The post-processor of <c>tokenizer.json</c>, if the vocabulary comes from it.</param>
        public GPT2Tokenizer(Dictionary<string, int> vocab, IEnumerable<(string, string)> merges, TokenizerConfig config = null,
            PostProcessor postProcessor = null) : base(config, new List<string> { "input_ids", "attention_mask" }) {
            SetSpecialTokens(unk: "<|endoftext|>", bos: "<|endoftext|>", eos: "<|endoftext|>");
            var backend = new Tokenizer(new BPE(vocab, merges, continuingSubwordPrefix: "", endOfWordSuffix: "")) {
                PreTokenizer = new PreTokenizers.ByteLevel(addPrefixSpace: Config.AddPrefixSpace ?? false),
                Decoder = new Decoders.ByteLevel(),
            };
            Initialize(backend, postProcessor);
        }

        /// <param name="vocabJson">The content of <c>vocab.json</c>.</param>
        /// <param name="mergesTxt">The content of <c>merges.txt</c>.</param>
        public GPT2Tokenizer(string vocabJson, string mergesTxt, TokenizerConfig config = null)
            : this(Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabJson), ReadMerges(mergesTxt), config) { }

        /// <summary>Takes the vocabulary and the post-processor of <c>tokenizer.json</c>.</summary>
        public static GPT2Tokenizer FromTokenizerJson(string tokenizerJson, TokenizerConfig config = null) {
            var (vocab, merges, postProcessor) = ReadBpe(tokenizerJson);
            return new GPT2Tokenizer(vocab, merges, config, postProcessor);
        }
    }
}
