using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The settings of <see cref="PreTrainedModel.Generate"/>, read from a model's <c>generation_config.json</c>.
    /// Has the fields of transformers' <c>GenerationConfig</c> that this package supports, with the same defaults.
    /// </summary>
    public class GenerationConfig {

        /// <summary>The maximum length of the sequence, prompt included. <see cref="MaxNewTokens"/> overrides it.</summary>
        [JsonProperty("max_length")]
        public int MaxLength { get; set; } = 20;

        /// <summary>The maximum number of tokens to generate, not counting the prompt.</summary>
        [JsonProperty("max_new_tokens")]
        public int? MaxNewTokens { get; set; }

        /// <summary>The minimum number of tokens to generate before an end-of-sequence token may end generation.</summary>
        [JsonProperty("min_new_tokens")]
        public int? MinNewTokens { get; set; }

        /// <summary>The maximum time to generate for, in seconds.</summary>
        [JsonProperty("max_time")]
        public float? MaxTime { get; set; }

        /// <summary>Whether to sample the next token. False picks the most likely one (greedy decoding).</summary>
        [JsonProperty("do_sample")]
        public bool DoSample { get; set; }

        /// <summary>Only 1 is supported: beam search is not.</summary>
        [JsonProperty("num_beams")]
        public int NumBeams { get; set; } = 1;

        [JsonProperty("use_cache")]
        public bool UseCache { get; set; } = true;

        [JsonProperty("temperature")]
        public float Temperature { get; set; } = 1f;

        [JsonProperty("top_k")]
        public int TopK { get; set; } = 50;

        [JsonProperty("top_p")]
        public float TopP { get; set; } = 1f;

        [JsonProperty("min_p")]
        public float? MinP { get; set; }

        [JsonProperty("repetition_penalty")]
        public float RepetitionPenalty { get; set; } = 1f;

        [JsonProperty("bos_token_id")]
        public int? BosTokenId { get; set; }

        /// <summary>The tokens that end generation. The json may have one id or a list.</summary>
        [JsonProperty("eos_token_id")]
        [JsonConverter(typeof(IdsConverter))]
        public int[] EosTokenId { get; set; }

        [JsonProperty("pad_token_id")]
        public int? PadTokenId { get; set; }

        public static GenerationConfig Deserialize(string json) {
            return JsonConvert.DeserializeObject<GenerationConfig>(json);
        }

        /// <summary>
        /// The generation config of a model without <c>generation_config.json</c>, from its <c>config.json</c>, as
        /// transformers' <c>GenerationConfig.from_model_config</c>: the fields of the config, and the fields that are
        /// still at their default from the text config of a model with a vision tower (for example the end-of-sequence
        /// token of Qwen3.5).
        /// </summary>
        public static GenerationConfig FromModelConfig(string configJson) {
            var config = JObject.Parse(configJson);
            var generationConfig = config.ToObject<GenerationConfig>();
            if (config["text_config"] is not JObject text) {
                return generationConfig;
            }
            var defaults = new GenerationConfig();
            var fromText = text.ToObject<GenerationConfig>();
            foreach (var property in typeof(GenerationConfig).GetProperties()) {
                var name = property.GetCustomAttributes(typeof(JsonPropertyAttribute), false).OfType<JsonPropertyAttribute>().FirstOrDefault()?.PropertyName;
                if (name == null || !text.ContainsKey(name)) {
                    continue;
                }
                if (JToken.DeepEquals(JToken.FromObject(property.GetValue(generationConfig) ?? JValue.CreateNull()),
                        JToken.FromObject(property.GetValue(defaults) ?? JValue.CreateNull()))) {
                    property.SetValue(generationConfig, property.GetValue(fromText));
                }
            }
            return generationConfig;
        }

        public GenerationConfig Clone() {
            var clone = (GenerationConfig)MemberwiseClone();
            clone.EosTokenId = (int[])EosTokenId?.Clone();
            return clone;
        }

        /// <summary>
        /// Reads one id or a list of ids.
        /// </summary>
        private class IdsConverter : JsonConverter<int[]> {
            public override int[] ReadJson(JsonReader reader, Type objectType, int[] existingValue, bool hasExistingValue, JsonSerializer serializer) {
                JToken token = JToken.Load(reader);
                return token.Type switch {
                    JTokenType.Null => null,
                    JTokenType.Array => token.Select(t => (int)t).ToArray(),
                    _ => new[] { (int)token },
                };
            }

            public override void WriteJson(JsonWriter writer, int[] value, JsonSerializer serializer) {
                serializer.Serialize(writer, value);
            }
        }
    }
}
