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
    }
}
