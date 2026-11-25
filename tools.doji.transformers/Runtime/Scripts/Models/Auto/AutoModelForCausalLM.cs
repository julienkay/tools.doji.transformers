using Doji.Downloads;
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Loads a pretrained model with a causal language modeling head, choosing the class from the
    /// <c>architectures</c> of its <c>config.json</c>.
    /// </summary>
    public static class AutoModelForCausalLM {

        /// <summary>
        /// Loads a model through AI Model Hub. Works on every platform.
        /// </summary>
        public static async Task<PreTrainedModel> FromPretrainedAsync(string modelId, BackendType backend = BackendType.GPUCompute,
            IProgress<DownloadProgress> progress = null, CancellationToken cancellationToken = default) {
            return await FromPretrainedAsync(await ModelRepository.OpenAsync(modelId, progress, cancellationToken), backend);
        }

        /// <summary>
        /// Loads a model through AI Model Hub without waiting. Works in the Editor and for bundled
        /// or already downloaded models on Windows, macOS, Linux, and iOS.
        /// </summary>
        public static PreTrainedModel FromPretrained(string modelId, BackendType backend = BackendType.GPUCompute) {
            return SyncTask.Run(FromPretrainedAsync(ModelRepository.Open(modelId), backend));
        }

        /// <summary>
        /// Loads a model from a GGUF file of llama.cpp, choosing the class from <c>general.architecture</c>. The
        /// tensors stay in the file until the worker needs them. F32 tensors stay float32, all other types become
        /// float16. Supported architectures: <c>qwen35</c> (<see cref="Qwen3_5ForCausalLM"/>).
        /// </summary>
        public static PreTrainedModel FromGguf(string path, BackendType backend = BackendType.GPUCompute) {
            return GgufModels.Load(path, backend);
        }

        private static async Task<PreTrainedModel> FromPretrainedAsync(ModelRepository files, BackendType backend) {
            var config = await files.ReadJsonAsync<PretrainedConfig>(PreTrainedModel.CONFIG_NAME);
            string architecture = config.Architectures?.Count > 0 ? config.Architectures[0]
                : throw new ArgumentException($"The config of '{files.ModelId}' names no architecture.");
            return architecture switch {
                "LlamaForCausalLM" => await PreTrainedModel.FromPretrainedAsync(files, backend, LlamaForCausalLM.Create),
                "Phi3ForCausalLM" => await PreTrainedModel.FromPretrainedAsync(files, backend, Phi3ForCausalLM.Create),
                // As transformers' AutoModelForCausalLM, the text model of a Qwen3.5 checkpoint with a vision tower.
                "Qwen3_5ForCausalLM" or "Qwen3_5ForConditionalGeneration"
                    => await PreTrainedModel.FromPretrainedAsync(files, backend, Qwen3_5ForCausalLM.Create),
                _ => throw new NotSupportedException($"The architecture '{architecture}' is not supported."),
            };
        }
    }
}
