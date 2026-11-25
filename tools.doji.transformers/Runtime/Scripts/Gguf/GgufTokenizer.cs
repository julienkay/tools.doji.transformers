using System;
using System.Collections.Generic;
using Doji.AI.Tokenizers;
using Doji.Gguf;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The tokenizer of a GGUF file of llama.cpp, from its <c>tokenizer.ggml.*</c> metadata: a <c>tokenizer.json</c>
    /// and a <see cref="TokenizerConfig"/>, as transformers' <c>convert_gguf_tokenizer</c> builds them.
    /// </summary>
    /// <remarks>
    /// The metadata holds the vocabulary, the merges, the token types and the special tokens, but not the
    /// pre-tokenizer: llama.cpp knows it by its name in <c>tokenizer.ggml.pre</c>. <see cref="PreTokenizers"/> maps
    /// each name to the normalizer, pre-tokenizer, post-processor and decoder of the <c>tokenizer.json</c> of the
    /// models that use it.
    /// </remarks>
    internal static class GgufTokenizer {

        private static JObject ByteLevel(bool useRegex = false, bool trimOffsets = false) =>
            new() { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = trimOffsets, ["use_regex"] = useRegex };

        private static JObject Split(string regex) => new() {
            ["type"] = "Split", ["pattern"] = new JObject { ["Regex"] = regex }, ["behavior"] = "Isolated", ["invert"] = false,
        };

        private static JObject Sequence(params JObject[] items) => new() { ["type"] = "Sequence", ["pretokenizers"] = new JArray(items) };

        private sealed class Parts {
            public JToken Normalizer, PreTokenizer, PostProcessor, Decoder;
            public bool SubwordAffixes;
        }

        /// <summary>The parts of <c>tokenizer.json</c> by the <c>tokenizer.ggml.pre</c> of byte-level BPE tokenizers.</summary>
        private static readonly Dictionary<string, Func<Parts>> PreTokenizers = new() {
            // Qwen3.5: letters and marks together.
            ["qwen35"] = () => new Parts {
                Normalizer = new JObject { ["type"] = "NFC" },
                PreTokenizer = Sequence(Split(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"), ByteLevel()),
                PostProcessor = ByteLevel(), Decoder = ByteLevel(), SubwordAffixes = true,
            },
            // Qwen2, Qwen2.5 and Qwen3.
            ["qwen2"] = () => new Parts {
                Normalizer = new JObject { ["type"] = "NFC" },
                PreTokenizer = Sequence(Split(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"), ByteLevel()),
                PostProcessor = ByteLevel(), Decoder = ByteLevel(), SubwordAffixes = true,
            },
            // SmolLM and SmolLM2: single digits, then the GPT-2 regex of ByteLevel.
            ["smollm"] = () => new Parts {
                PreTokenizer = Sequence(new JObject { ["type"] = "Digits", ["individual_digits"] = true }, ByteLevel(useRegex: true, trimOffsets: true)),
                Decoder = ByteLevel(useRegex: true, trimOffsets: true),
            },
        };

        /// <summary>Whether this package can build the tokenizer of <paramref name="file"/>.</summary>
        public static bool IsSupported(GgufFile file) =>
            file.GetValue("tokenizer.ggml.model", "") == "gpt2" && PreTokenizers.ContainsKey(file.GetValue("tokenizer.ggml.pre", ""));

        /// <summary>The <c>tokenizer.json</c> of the tokenizer of <paramref name="file"/>.</summary>
        public static string TokenizerJson(GgufFile file) {
            string model = file.GetValue("tokenizer.ggml.model", "");
            string pre = file.GetValue("tokenizer.ggml.pre", "");
            if (model != "gpt2") {
                throw new NotSupportedException($"{file.Path} has a tokenizer of type '{model}'. Supported: gpt2 (byte-level BPE).");
            }
            if (!PreTokenizers.TryGetValue(pre, out var create)) {
                throw new NotSupportedException($"{file.Path} has the pre-tokenizer '{pre}'. Supported: {string.Join(", ", PreTokenizers.Keys)}.");
            }
            var parts = create();
            string[] tokens = file.GetValue<string[]>("tokenizer.ggml.tokens");
            int[] types = file.TryGetValue("tokenizer.ggml.token_type", out int[] t) ? t : null;
            var vocab = new JObject();
            var added = new JArray();
            for (int i = 0; i < tokens.Length; i++) {
                // llama_token_type: 1 normal, 2 unknown, 3 control (special), 4 user defined, 5 unused, 6 byte.
                int type = types?[i] ?? 1;
                if (type == 5) {
                    continue;
                }
                vocab[tokens[i]] = i;
                if (type == 3 || type == 4) {
                    added.Add(new JObject {
                        ["id"] = i, ["content"] = tokens[i], ["single_word"] = false, ["lstrip"] = false, ["rstrip"] = false,
                        ["normalized"] = type == 4, ["special"] = type == 3,
                    });
                }
            }
            var json = new JObject {
                ["version"] = "1.0",
                ["truncation"] = null,
                ["padding"] = null,
                ["added_tokens"] = added,
                ["normalizer"] = parts.Normalizer,
                ["pre_tokenizer"] = parts.PreTokenizer,
                ["post_processor"] = parts.PostProcessor,
                ["decoder"] = parts.Decoder,
                ["model"] = new JObject {
                    ["type"] = "BPE", ["dropout"] = null, ["unk_token"] = null,
                    ["continuing_subword_prefix"] = parts.SubwordAffixes ? "" : null,
                    ["end_of_word_suffix"] = parts.SubwordAffixes ? "" : null,
                    ["fuse_unk"] = false, ["byte_fallback"] = false, ["ignore_merges"] = false,
                    ["vocab"] = vocab,
                    ["merges"] = new JArray(file.GetValue<string[]>("tokenizer.ggml.merges")),
                },
            };
            return json.ToString(Formatting.None);
        }

        /// <summary>
        /// The special tokens, <c>add_bos_token</c>, <c>add_eos_token</c> and the chat template of the tokenizer of
        /// <paramref name="file"/>.
        /// </summary>
        public static TokenizerConfig Config(GgufFile file) {
            string[] tokens = file.GetValue<string[]>("tokenizer.ggml.tokens");
            AddedToken Token(string key) =>
                file.TryGetValue($"tokenizer.ggml.{key}", out int id) && id >= 0 && id < tokens.Length ? new AddedToken(tokens[id], special: true) : null;
            var config = new TokenizerConfig {
                BosToken = Token("bos_token_id"),
                EosToken = Token("eos_token_id"),
                UnkToken = Token("unknown_token_id"),
                PadToken = Token("padding_token_id"),
            };
            // As transformers, only a file that asks for them adds the BOS and EOS tokens to each encoding.
            if (file.GetValue("tokenizer.ggml.add_bos_token", false)) {
                config.AddBosToken = true;
            }
            if (file.GetValue("tokenizer.ggml.add_eos_token", false)) {
                config.AddEosToken = true;
            }
            if (file.TryGetValue("tokenizer.chat_template", out string template)) {
                config.ChatTemplate = template;
            }
            return config;
        }
    }
}
