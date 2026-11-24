using Doji.Downloads;
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A Llama model with a language modeling head, for text generation.
    /// </summary>
    public class LlamaForCausalLM : PreTrainedModel {

        /// <inheritdoc cref="PreTrainedModel(Model, PretrainedConfig, GenerationConfig, BackendType)"/>
        public LlamaForCausalLM(Model model, PretrainedConfig config, GenerationConfig generationConfig = null, BackendType backend = BackendType.GPUCompute)
            : base(model, config, generationConfig, backend) { }

        /// <inheritdoc cref="PreTrainedModel(ModelFile, PretrainedConfig, GenerationConfig, BackendType)"/>
        public LlamaForCausalLM(ModelFile file, PretrainedConfig config, GenerationConfig generationConfig = null, BackendType backend = BackendType.GPUCompute)
            : base(file, config, generationConfig, backend) { }

        /// <summary>
        /// Loads a pretrained model through AI Model Hub. Works on every platform.
        /// </summary>
        public static async Task<LlamaForCausalLM> FromPretrainedAsync(string modelId, BackendType backend = BackendType.GPUCompute,
            IProgress<DownloadProgress> progress = null, CancellationToken cancellationToken = default) {
            return await FromPretrainedAsync(await ModelRepository.OpenAsync(modelId, progress, cancellationToken), backend, Create);
        }

        /// <summary>
        /// Loads a pretrained model through AI Model Hub without waiting. Works in the Editor and for bundled
        /// or already downloaded models on Windows, macOS, Linux, and iOS.
        /// </summary>
        public static LlamaForCausalLM FromPretrained(string modelId, BackendType backend = BackendType.GPUCompute) {
            return SyncTask.Run(FromPretrainedAsync(ModelRepository.Open(modelId), backend, Create));
        }

        internal static LlamaForCausalLM Create(ModelFile file, PretrainedConfig config, GenerationConfig generationConfig, BackendType backend) {
            return new LlamaForCausalLM(file, config, generationConfig, backend);
        }
    }
}
