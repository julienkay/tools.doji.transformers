using System;
using Doji.Gguf;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Creates models from GGUF files of llama.cpp.
    /// </summary>
    internal static class GgufModels {

        /// <summary>
        /// Opens the GGUF file at <paramref name="path"/>, builds the model of its architecture, and creates the
        /// model class. The weights go from the file to the worker one by one.
        /// </summary>
        public static PreTrainedModel Load(string path, BackendType backend, string expectedArchitecture = null) {
            var file = GgufFile.Open(path);
            string architecture = file.Architecture;
            if (expectedArchitecture != null && architecture != expectedArchitecture) {
                throw new ArgumentException($"{path} holds a model of architecture '{architecture}', not '{expectedArchitecture}'.");
            }
            switch (architecture) {
                case Qwen3_5Gguf.Architecture: {
                    var config = Qwen3_5Gguf.Config(file);
                    using var model = Qwen3_5Gguf.Build(file, config);
                    return new Qwen3_5ForCausalLM(model, config, GenerationConfig(file), backend);
                }
                default:
                    throw new NotSupportedException($"GGUF files of architecture '{architecture}' are not supported. Supported: {Qwen3_5Gguf.Architecture}.");
            }
        }

        /// <summary>
        /// The special tokens of the tokenizer metadata, as transformers takes them from a GGUF file.
        /// </summary>
        public static GenerationConfig GenerationConfig(GgufFile file) {
            var config = new GenerationConfig();
            if (file.TryGetValue("tokenizer.ggml.bos_token_id", out int bos)) {
                config.BosTokenId = bos;
            }
            if (file.TryGetValue("tokenizer.ggml.eos_token_id", out int eos)) {
                config.EosTokenId = new[] { eos };
            }
            if (file.TryGetValue("tokenizer.ggml.padding_token_id", out int pad)) {
                config.PadTokenId = pad;
            }
            return config;
        }
    }
}
