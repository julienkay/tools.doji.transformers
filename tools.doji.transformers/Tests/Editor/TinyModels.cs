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

        /// <summary>
        /// The environment variable with the folder of the tiny Qwen3.5 model as GGUF files of llama.cpp:
        /// model-f32.gguf, model-bf16.gguf and model-q8_0.gguf of julienkay/tiny-random-Qwen3_5ForCausalLM-GGUF.
        /// AI Model Hub does not serve GGUF files yet.
        /// </summary>
        public const string Qwen3_5GgufVariable = "DOJI_TINY_QWEN3_5_GGUF";

        /// <summary>
        /// The environment variable with the folder of model-f32.gguf, model-bf16.gguf and model-q8_0.gguf of the tiny
        /// Llama model (<see cref="Llama"/>), from convert_hf_to_gguf.py of llama.cpp:
        /// julienkay/tiny-random-LlamaForCausalLM-GGUF.
        /// </summary>
        public const string LlamaGgufVariable = "DOJI_TINY_LLAMA_GGUF";

        /// <summary>
        /// The environment variable with the folder of the GGUF files of the tiny Phi-3 model (<see cref="Phi3"/>):
        /// julienkay/tiny-random-Phi3ForCausalLM-GGUF.
        /// </summary>
        public const string Phi3GgufVariable = "DOJI_TINY_PHI3_GGUF";

        /// <summary>
        /// The environment variable with the folder of the GGUF files of a 2-layer random-weight Qwen3ForCausalLM with
        /// the tokenizer of Qwen/Qwen3.5-0.8B (tools/causal_lm/make_models.py tiny --arch qwen3):
        /// julienkay/tiny-random-Qwen3ForCausalLM-GGUF.
        /// </summary>
        public const string Qwen3GgufVariable = "DOJI_TINY_QWEN3_GGUF";
    }
}
