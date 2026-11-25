using Doji.AI.Tokenizers;
using Doji.AI.Tokenizers.Models;
using Doji.AI.Tokenizers.PostProcessors;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Decoders = Doji.AI.Tokenizers.Decoders;
using PreTokenizers = Doji.AI.Tokenizers.PreTokenizers;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The sentencepiece-style BPE tokenizer of Llama, as transformers' <c>LlamaTokenizer</c> builds it: a metaspace
    /// pre-tokenizer and byte fallback.
    /// </summary>
    public class LlamaTokenizer : TokenizersBackend {

        private const char Underline = '▁';

        /// <param name="vocab">The vocabulary.</param>
        /// <param name="merges">The merges by rank.</param>
        /// <param name="postProcessor">The post-processor of <c>tokenizer.json</c>, if the vocabulary comes from it.</param>
        public LlamaTokenizer(Dictionary<string, int> vocab, IEnumerable<(string, string)> merges, TokenizerConfig config = null,
            PostProcessor postProcessor = null) : base(config, new List<string> { "input_ids", "attention_mask" }, Side.Left) {
            SetSpecialTokens(unk: "<unk>", bos: "<s>", eos: "</s>");
            bool addPrefixSpace = Config.AddPrefixSpace ?? true;
            bool legacy = Config.Legacy ?? false;
            var prependScheme = !addPrefixSpace ? PreTokenizers.PrependScheme.Never
                : legacy ? PreTokenizers.PrependScheme.Always : PreTokenizers.PrependScheme.First;
            var decoders = new List<Decoders.Decoder> {
                new Decoders.Replace(Tokenizers.Normalizers.Pattern.String(Underline.ToString()), " "),
                new Decoders.ByteFallback(),
                new Decoders.Fuse(),
            };
            if (addPrefixSpace) {
                decoders.Add(new Decoders.Strip(' ', 1, 0));
            }
            var backend = new Tokenizer(new BPE(vocab, merges, fuseUnk: true, byteFallback: true)) {
                PreTokenizer = new PreTokenizers.Metaspace(Underline, prependScheme, split: false),
                Decoder = new Decoders.Sequence(decoders),
            };
            Initialize(backend, postProcessor);
        }

        /// <summary>Takes the vocabulary and the post-processor of <c>tokenizer.json</c>.</summary>
        public static LlamaTokenizer FromTokenizerJson(string tokenizerJson, TokenizerConfig config = null) {
            var (vocab, merges, postProcessor) = ReadBpe(tokenizerJson);
            return new LlamaTokenizer(vocab, merges, config, postProcessor);
        }

        /// <summary>
        /// Takes the vocabulary of a sentencepiece model (<c>tokenizer.model</c>), and makes the merges from it, as
        /// transformers does. Its control and user defined pieces become extra special tokens.
        /// </summary>
        public static LlamaTokenizer FromSentencePiece(byte[] model, TokenizerConfig config = null) {
            config ??= new TokenizerConfig();
            ModelProto proto = ModelProto.Parse(model);
            var vocab = new Dictionary<string, int>();
            for (int id = 0; id < proto.Pieces.Count; id++) {
                vocab[proto.Pieces[id].Piece] = id;
            }
            for (int id = 0; id < proto.Pieces.Count; id++) {
                var piece = proto.Pieces[id];
                if (piece.Type is ModelProto.PieceType.Control or ModelProto.PieceType.UserDefined) {
                    config.ExtraSpecialTokens.Add(new AddedToken(piece.Piece, special: piece.Type == ModelProto.PieceType.Control, normalized: false));
                }
            }
            return new LlamaTokenizer(vocab, GenerateMerges(vocab), config);
        }

        public static LlamaTokenizer FromSentencePiece(string path, TokenizerConfig config = null) => FromSentencePiece(File.ReadAllBytes(path), config);

        /// <summary>
        /// transformers' <c>generate_merges</c>: each split of a piece into two pieces of the vocabulary is a merge,
        /// ranked by the id of the piece.
        /// </summary>
        internal static List<(string, string)> GenerateMerges(Dictionary<string, int> vocab) {
            var merges = new List<(string Left, string Right, int Score)>();
            foreach (var entry in vocab) {
                string merge = entry.Key;
                var local = new List<(string, string, int)>();
                for (int index = 1; index < merge.Length; index++) {
                    string left = merge[..index], right = merge[index..];
                    if (vocab.ContainsKey(left) && vocab.ContainsKey(right)) {
                        local.Add((left, right, entry.Value));
                    }
                }
                merges.AddRange(local.OrderBy(m => vocab[m.Item1]).ThenBy(m => vocab[m.Item2]));
            }
            return merges.OrderBy(m => m.Score).ThenBy(m => CodePoints(m.Left)).ThenBy(m => CodePoints(m.Right))
                .Select(m => (m.Left, m.Right)).ToList();
        }

        /// <summary>
        /// transformers' <c>generate_merges</c> with scores: each split of a piece into two pieces of the vocabulary is
        /// a merge, ranked by the score of the piece, the highest first. A GGUF file of a sentencepiece tokenizer has
        /// the scores of its pieces, and no merges.
        /// </summary>
        /// <param name="scores">The score of each piece, in the order of the pieces.</param>
        internal static List<(string, string)> GenerateMerges(Dictionary<string, int> vocab, IEnumerable<(string Piece, float Score)> scores) {
            var merges = new List<(string Left, string Right, float Score)>();
            foreach (var (merge, score) in scores) {
                var local = new List<(string, string, float)>();
                for (int index = 1; index < merge.Length; index++) {
                    string left = merge[..index], right = merge[index..];
                    if (vocab.ContainsKey(left) && vocab.ContainsKey(right)) {
                        local.Add((left, right, score));
                    }
                }
                merges.AddRange(local.OrderBy(m => vocab[m.Item1]).ThenBy(m => vocab[m.Item2]));
            }
            // As Python's sorted(..., reverse=True): descending on the whole key, and stable.
            return merges.OrderByDescending(m => m.Score).ThenByDescending(m => CodePoints(m.Left)).ThenByDescending(m => CodePoints(m.Right))
                .Select(m => (m.Left, m.Right)).ToList();
        }

        private static int CodePoints(string text) => text.Length - text.Count(char.IsLowSurrogate);
    }
}
