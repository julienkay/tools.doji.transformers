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
