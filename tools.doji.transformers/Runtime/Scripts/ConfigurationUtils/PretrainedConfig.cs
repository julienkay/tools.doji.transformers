using Newtonsoft.Json;
using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Base class for all configuration classes. Handles a few parameters common to all models' configurations
    /// as well as methods for loading configurations.
    /// </summary>
    public class PretrainedConfig {

        /* Common attributes(present in all subclasses): */

        [JsonProperty("vocab_size")]
        public int VocabSize { get; set; }

        [JsonProperty("hidden_size")]
        public int HiddenSize { get; set; }

        [JsonProperty("num_attention_heads")]
        public int NumAttentionHeads { get; set; }

        [JsonProperty("num_hidden_layers")]
        public int NumHiddenLayers { get; set; }

        /// <summary>
        /// The number of key and value heads of grouped-query attention. Null means <see cref="NumAttentionHeads"/>.
        /// </summary>
        [JsonProperty("num_key_value_heads")]
        public int? NumKeyValueHeads { get; set; }

        /// <summary>
        /// The size of one attention head. Null means <see cref="HiddenSize"/> / <see cref="NumAttentionHeads"/>.
        /// </summary>
        [JsonProperty("head_dim")]
        public int? HeadDim { get; set; }


        [JsonProperty("_name_or_path")]
        public string NameOrPath { get; set; }

        [JsonProperty("architectures")]
        public List<string> Architectures { get; set; }

        [JsonProperty("is_encoder_decoder")]
        public bool IsEncoderDecoder { get; set; }


        /* These are actually not part of the base config class, but still convenient to have here */

        [JsonProperty("sliding_window")]
        public int? SlidingWindow { get; set; }

        [JsonProperty("max_position_embeddings")]
        public int? MaxPositionEmbeddings { get; set; }

        [JsonProperty("model_type")]
        public string ModelType { get; set; }

        /// <summary>
        /// The config of the language model, in the config of a model that also has a vision tower, such as
        /// Qwen3.5. Null when the config is that of the language model.
        /// </summary>
        [JsonProperty("text_config")]
        public PretrainedConfig TextConfig { get; set; }

        /// <summary>
        /// The kind of each layer of a model that mixes layer kinds: <c>"full_attention"</c> or
        /// <c>"linear_attention"</c>. Null means all layers are full attention.
        /// </summary>
        [JsonProperty("layer_types")]
        public List<string> LayerTypes { get; set; }

        /* The linear attention layers of Gated DeltaNet (Qwen3.5). */

        [JsonProperty("linear_conv_kernel_dim")]
        public int? LinearConvKernelDim { get; set; }

        [JsonProperty("linear_key_head_dim")]
        public int? LinearKeyHeadDim { get; set; }

        [JsonProperty("linear_value_head_dim")]
        public int? LinearValueHeadDim { get; set; }

        [JsonProperty("linear_num_key_heads")]
        public int? LinearNumKeyHeads { get; set; }

        [JsonProperty("linear_num_value_heads")]
        public int? LinearNumValueHeads { get; set; }

        /// <summary>
        /// The config of the language model, as transformers' <c>get_text_config</c>: <see cref="TextConfig"/>, or
        /// this config when it has none.
        /// </summary>
        public PretrainedConfig GetTextConfig() => TextConfig ?? this;

        /// <summary>
        /// Whether layer <paramref name="layerIdx"/> is a linear attention layer.
        /// </summary>
        public bool IsLinearAttention(int layerIdx) => LayerTypes != null && LayerTypes[layerIdx] == "linear_attention";

        public PretrainedConfig() {
            IsEncoderDecoder = false;
        }
    }
}