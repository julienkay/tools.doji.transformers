using Doji.AI.Tokenizers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The content of <c>tokenizer_config.json</c>: the special tokens and the settings of a tokenizer.
    /// </summary>
    public class TokenizerConfig {

        [JsonProperty("tokenizer_class")]
        public string TokenizerClass { get; set; }

        [JsonProperty("add_prefix_space")]
        public bool? AddPrefixSpace { get; set; }

        [JsonProperty("add_bos_token")]
        public bool? AddBosToken { get; set; }

        [JsonProperty("add_eos_token")]
        public bool? AddEosToken { get; set; }

        [JsonProperty("legacy")]
        public bool? Legacy { get; set; }

        /// <summary>The added tokens by id.</summary>
        [JsonProperty("added_tokens_decoder")]
        [JsonConverter(typeof(AddedTokensConverter))]
        public Dictionary<int, AddedToken> AddedTokensDecoder { get; set; }

        [JsonProperty("bos_token")]
        [JsonConverter(typeof(AddedTokenConverter))]
        public AddedToken BosToken { get; set; }

        [JsonProperty("eos_token")]
        [JsonConverter(typeof(AddedTokenConverter))]
        public AddedToken EosToken { get; set; }

        [JsonProperty("unk_token")]
        [JsonConverter(typeof(AddedTokenConverter))]
        public AddedToken UnkToken { get; set; }

        [JsonProperty("sep_token")]
        [JsonConverter(typeof(AddedTokenConverter))]
        public AddedToken SepToken { get; set; }

        [JsonProperty("pad_token")]
        [JsonConverter(typeof(AddedTokenConverter))]
        public AddedToken PadToken { get; set; }

        [JsonProperty("cls_token")]
        [JsonConverter(typeof(AddedTokenConverter))]
        public AddedToken ClsToken { get; set; }

        [JsonProperty("mask_token")]
        [JsonConverter(typeof(AddedTokenConverter))]
        public AddedToken MaskToken { get; set; }

        /// <summary>
        /// The special tokens beyond the named ones: <c>extra_special_tokens</c> (transformers v5) or
        /// <c>additional_special_tokens</c> (v4).
        /// </summary>
        [JsonIgnore]
        public List<AddedToken> ExtraSpecialTokens { get; set; } = new();

        [JsonProperty("extra_special_tokens")]
        private JToken ExtraSpecialTokensJson { set => ExtraSpecialTokens.AddRange(ReadTokenList(value)); }

        [JsonProperty("additional_special_tokens")]
        private JToken AdditionalSpecialTokensJson { set => ExtraSpecialTokens.AddRange(ReadTokenList(value)); }

        [JsonProperty("model_input_names")]
        public List<string> ModelInputNames { get; set; }

        [JsonProperty("clean_up_tokenization_spaces")]
        public bool? CleanUpTokenizationSpaces { get; set; }

        [JsonProperty("split_special_tokens")]
        public bool? SplitSpecialTokens { get; set; }

        /// <summary>The longest input the model takes. Many configs write a huge number when there is no limit.</summary>
        [JsonIgnore]
        public int? ModelMaxLength { get; set; }

        [JsonProperty("model_max_length")]
        private JToken ModelMaxLengthJson {
            set => ModelMaxLength = value == null || value.Type == JTokenType.Null ? null
                : (int)Math.Min(double.Parse(value.ToString(), System.Globalization.CultureInfo.InvariantCulture), int.MaxValue);
        }

        [JsonProperty("padding_side")]
        public Side? PaddingSide { get; set; }

        [JsonProperty("truncation_side")]
        public Side? TruncationSide { get; set; }

        /// <summary>The Jinja chat template: a string, or a list of named templates in some models.</summary>
        [JsonProperty("chat_template")]
        public JToken ChatTemplate { get; set; }

        public static TokenizerConfig Deserialize(string json) {
            return JsonConvert.DeserializeObject<TokenizerConfig>(json) ?? new TokenizerConfig();
        }

        /// <summary>
        /// Reads a special token: a string, or an object with the fields of <see cref="AddedToken"/>.
        /// A named special token is special even when its object says otherwise, as in transformers.
        /// </summary>
        internal static AddedToken ReadToken(JToken json, bool special = true) {
            if (json == null || json.Type == JTokenType.Null) {
                return null;
            }
            if (json.Type == JTokenType.String) {
                return new AddedToken((string)json, special: special);
            }
            return new AddedToken(
                (string)json["content"],
                special: special || ((bool?)json["special"] ?? false),
                normalized: (bool?)json["normalized"],
                singleWord: (bool?)json["single_word"] ?? false,
                lstrip: (bool?)json["lstrip"] ?? false,
                rstrip: (bool?)json["rstrip"] ?? false);
        }

        private static IEnumerable<AddedToken> ReadTokenList(JToken json) {
            return json is JArray array ? array.Select(t => ReadToken(t)).Where(t => t != null) : Enumerable.Empty<AddedToken>();
        }

        private class AddedTokenConverter : JsonConverter {
            public override bool CanConvert(Type objectType) => objectType == typeof(AddedToken);
            public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer) {
                return ReadToken(JToken.Load(reader));
            }
            public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer) {
                throw new NotSupportedException();
            }
        }

        private class AddedTokensConverter : JsonConverter {
            public override bool CanConvert(Type objectType) => objectType == typeof(Dictionary<int, AddedToken>);
            public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer) {
                if (JToken.Load(reader) is not JObject json) {
                    return null;
                }
                return json.Properties().ToDictionary(p => int.Parse(p.Name), p => ReadToken(p.Value, special: false));
            }
            public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer) {
                throw new NotSupportedException();
            }
        }
    }
}
