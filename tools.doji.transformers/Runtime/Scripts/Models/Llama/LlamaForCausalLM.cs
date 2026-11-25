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

        /// <summary>
        /// Loads a model from a GGUF file of llama.cpp (architecture <c>llama</c>). The tensors stay in the file
        /// until the worker needs them. Q8_0, Q4_K, Q5_K and Q6_K weights stay quantized on the device, F32 tensors
        /// stay float32, and other types become float16.
        /// </summary>
        public static LlamaForCausalLM FromGguf(string path, BackendType backend = BackendType.GPUCompute) {
            return (LlamaForCausalLM)GgufModels.Load(path, backend, LlamaGguf.Llama);
        }

        internal static LlamaForCausalLM Create(ModelFile file, PretrainedConfig config, GenerationConfig generationConfig, BackendType backend) {
            return new LlamaForCausalLM(file, config, generationConfig, backend);
        }
    }
}
