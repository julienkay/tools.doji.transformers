namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// Tiny models with random weights, for regression tests against the transformers library.
    /// The companion project's model-hub manifest pins their revisions.
    /// tools/causal_lm/make_models.py makes them and writes the reference data in Resources.
    /// </summary>
    public static class TinyModels {

        /// <summary>
        /// A 2-layer random-weight LlamaForCausalLM with grouped-query attention and the tokenizer of
        /// HuggingFaceTB/SmolLM2-135M-Instruct, exported to plain ONNX.
        /// </summary>
        public const string Llama = "julienkay/tiny-random-LlamaForCausalLM-onnx";

        /// <summary>
        /// A 2-layer random-weight Phi3ForCausalLM with the tokenizer of microsoft/Phi-3-mini-4k-instruct,
        /// exported to plain ONNX.
        /// </summary>
        public const string Phi3 = "julienkay/tiny-random-Phi3ForCausalLM-onnx";

        /// <summary>
        /// A 4-layer random-weight Qwen3_5ForCausalLM (three linear attention layers, then one full attention
        /// layer) with the tokenizer of Qwen/Qwen3.5-0.8B, exported to ONNX with the GatedDeltaRule node of ai.doji.
        /// </summary>
        public const string Qwen3_5 = "julienkay/tiny-random-Qwen3_5ForCausalLM-onnx";
    }
}
