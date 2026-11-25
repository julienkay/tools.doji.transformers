using Doji.Downloads;
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A Qwen3.5 language model with a language modeling head, for text generation. Its layers mix full attention
    /// with the linear attention of Gated DeltaNet, which needs tools.doji.sentis-utils (<c>GatedDeltaRule</c>).
    /// It also loads the text model of a Qwen3.5 checkpoint with a vision tower (Qwen3_5ForConditionalGeneration).
    /// </summary>
    public class Qwen3_5ForCausalLM : PreTrainedModel {

        /// <inheritdoc cref="PreTrainedModel(Model, PretrainedConfig, GenerationConfig, BackendType)"/>
        public Qwen3_5ForCausalLM(Model model, PretrainedConfig config, GenerationConfig generationConfig = null, BackendType backend = BackendType.GPUCompute)
            : base(model, config, generationConfig, backend) { }

        /// <inheritdoc cref="PreTrainedModel(ModelFile, PretrainedConfig, GenerationConfig, BackendType)"/>
        public Qwen3_5ForCausalLM(ModelFile file, PretrainedConfig config, GenerationConfig generationConfig = null, BackendType backend = BackendType.GPUCompute)
            : base(file, config, generationConfig, backend) { }

        /// <summary>
        /// Loads a pretrained model through AI Model Hub. Works on every platform.
        /// </summary>
        public static async Task<Qwen3_5ForCausalLM> FromPretrainedAsync(string modelId, BackendType backend = BackendType.GPUCompute,
            IProgress<DownloadProgress> progress = null, CancellationToken cancellationToken = default) {
            return await FromPretrainedAsync(await ModelRepository.OpenAsync(modelId, progress, cancellationToken), backend, Create);
        }

        /// <summary>
        /// Loads a pretrained model through AI Model Hub without waiting. Works in the Editor and for bundled
        /// or already downloaded models on Windows, macOS, Linux, and iOS.
        /// </summary>
        public static Qwen3_5ForCausalLM FromPretrained(string modelId, BackendType backend = BackendType.GPUCompute) {
            return SyncTask.Run(FromPretrainedAsync(ModelRepository.Open(modelId), backend, Create));
        }

        /// <summary>
        /// Loads a model from a GGUF file of llama.cpp (architecture <c>qwen35</c>). The tensors stay in the file
        /// until the worker needs them. F32 tensors stay float32, all other types become float16.
        /// </summary>
        public static Qwen3_5ForCausalLM FromGguf(string path, BackendType backend = BackendType.GPUCompute) {
            return (Qwen3_5ForCausalLM)GgufModels.Load(path, backend, Qwen3_5Gguf.Architecture);
        }

        internal static Qwen3_5ForCausalLM Create(ModelFile file, PretrainedConfig config, GenerationConfig generationConfig, BackendType backend) {
            return new Qwen3_5ForCausalLM(file, config, generationConfig, backend);
        }
    }
}
