using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The base class of the models: a decoder-only language model in one ONNX file, run with Unity Inference Engine.
    /// </summary>
    /// <remarks>
    /// The ONNX model runs both the prompt and each new token, with the inputs and outputs of optimum exports:
    /// <c>input_ids</c>, <c>attention_mask</c>, <c>position_ids</c> and <c>past_key_values.{i}.key/value</c> in,
    /// <c>logits</c> and <c>present.{i}.key/value</c> out. tools/causal_lm/make_models.py exports models in this form.
    /// </remarks>
    public abstract partial class PreTrainedModel : IDisposable {

        public const string MODEL_NAME = "model.onnx";

        /// <summary>
        /// The model converted to Inference Engine's format. Repositories with this file load it without conversion.
        /// </summary>
        public const string SENTIS_MODEL_NAME = "model.sentis";
        public const string CONFIG_NAME = "config.json";
        public const string GENERATION_CONFIG_NAME = "generation_config.json";

        /// <summary>
        /// The Hugging Face repository id the model was loaded from.
        /// </summary>
        public string ModelId { get; internal set; }

        public PretrainedConfig Config { get; }

        /// <summary>
        /// The default settings of <see cref="Generate"/>, from <c>generation_config.json</c>.
        /// </summary>
        public GenerationConfig GenerationConfig { get; set; }

        private readonly Worker _worker;

        /// <summary>
        /// On <see cref="BackendType.GPUCompute"/>, the command buffer the worker records into. <see cref="Run(Tensor{int}, DynamicCache)"/>
        /// executes it after each run, and <see cref="RunInParts"/> after each part. Null on the CPU backend.
        /// </summary>
        private readonly CommandBuffer _commandBuffer;
        private readonly int _numKeyValueHeads, _headDim;
        private readonly int _inputIds, _attentionMask, _positionIds, _logits;
        private readonly int[] _pastInputs, _presentOutputs;

        /// <summary>
        /// Creates the model from a model in memory. On <see cref="BackendType.GPUCompute"/> its weights move to the
        /// GPU, so <paramref name="model"/> cannot create another worker afterwards.
        /// </summary>
        protected PreTrainedModel(Model model, PretrainedConfig config, GenerationConfig generationConfig, BackendType backend)
            : this(model?.inputs, model?.outputs, config, generationConfig) {
            // A uint8 embedding table otherwise dequantizes as a whole on every step.
            QuantizedGather.Fuse(model);
            _worker = WorkerUtils.CreateWorkerWithWeightsOnGpu(model, backend);
            _commandBuffer = CreateCommandBuffer(backend);
        }

        /// <summary>
        /// Creates the model from an open .sentis file. The weights go from the file to the worker, so they are
        /// never in managed memory. The file can be disposed afterwards.
        /// </summary>
        protected PreTrainedModel(ModelFile file, PretrainedConfig config, GenerationConfig generationConfig, BackendType backend)
            : this(file?.Inputs, file?.Outputs, config, generationConfig) {
            // Like transformers and vLLM, a float16 model keeps its KV cache in float16, which halves its memory.
            if (backend == BackendType.GPUCompute && file.HasHalfPrecisionWeights) {
                int layers = config.NumHiddenLayers;
                var past = new List<string>();
                var present = new List<string>();
                for (int i = 0; i < layers; i++) {
                    past.Add($"past_key_values.{i}.key");
                    past.Add($"past_key_values.{i}.value");
                    present.Add($"present.{i}.key");
                    present.Add($"present.{i}.value");
                }
                if (file.UseHalfPrecision(past, present) != 4 * layers) {
                    throw new InvalidOperationException("The KV cache inputs and outputs of the model are not all float32.");
                }
                CacheDataType = DataType.Short;
            }
            _worker = file.CreateWorker(backend);
            _commandBuffer = CreateCommandBuffer(backend);
        }

        private static CommandBuffer CreateCommandBuffer(BackendType backend) {
            return backend == BackendType.GPUCompute ? new CommandBuffer { name = "Doji.Transformers" } : null;
        }

        /// <summary>
        /// The data type of the KV cache: <see cref="DataType.Short"/> (float16) for a model with float16 weights on
        /// <see cref="BackendType.GPUCompute"/>, otherwise <see cref="DataType.Float"/>.
        /// </summary>
        public DataType CacheDataType { get; } = DataType.Float;

        private PreTrainedModel(IReadOnlyList<Model.Input> modelInputs, IReadOnlyList<Model.Output> modelOutputs,
            PretrainedConfig config, GenerationConfig generationConfig) {
            if (modelInputs == null) {
                throw new ArgumentNullException("model");
            }
            Config = config;
            GenerationConfig = generationConfig ?? new GenerationConfig();
            _numKeyValueHeads = config.NumKeyValueHeads ?? config.NumAttentionHeads;
            _headDim = config.HeadDim ?? config.HiddenSize / config.NumAttentionHeads;

            // Inputs and outputs by index, not by name: setting inputs by name fails on quantized models.
            var inputs = modelInputs.Select(i => i.name).ToList();
            var outputs = modelOutputs.Select(o => o.name).ToList();
            int Input(string name) => IndexOf(inputs, name, "input");
            int Output(string name) => IndexOf(outputs, name, "output");
            _inputIds = Input("input_ids");
            _attentionMask = Input("attention_mask");
            _positionIds = Input("position_ids");
            _logits = Output("logits");
            int layers = config.NumHiddenLayers;
            _pastInputs = new int[2 * layers];
            _presentOutputs = new int[2 * layers];
            for (int i = 0; i < layers; i++) {
                _pastInputs[2 * i] = Input($"past_key_values.{i}.key");
                _pastInputs[2 * i + 1] = Input($"past_key_values.{i}.value");
                _presentOutputs[2 * i] = Output($"present.{i}.key");
                _presentOutputs[2 * i + 1] = Output($"present.{i}.value");
            }
        }

        private static int IndexOf(List<string> names, string name, string kind) {
            int index = names.IndexOf(name);
            return index >= 0 ? index : throw new ArgumentException(
                $"The model has no {kind} '{name}'. Export it with tools/causal_lm/make_models.py.");
        }

        /// <summary>
        /// Reads the configs and the model of a repository and creates the model class.
        /// </summary>
        internal static async Task<T> FromPretrainedAsync<T>(ModelRepository files, BackendType backend,
            Func<ModelFile, PretrainedConfig, GenerationConfig, BackendType, T> create) where T : PreTrainedModel {
            var config = await files.ReadJsonAsync<PretrainedConfig>(CONFIG_NAME);
            // Like transformers' GenerationConfig.from_model_config: config.json has the special token ids too.
            var generationConfig = GenerationConfig.Deserialize(await files.ReadTextAsync(
                files.Has(GENERATION_CONFIG_NAME) ? GENERATION_CONFIG_NAME : CONFIG_NAME));
            using var file = await files.OpenModelAsync(files.Has(SENTIS_MODEL_NAME) ? SENTIS_MODEL_NAME : MODEL_NAME);
            T result = create(file, config, generationConfig, backend);
            result.ModelId = files.ModelId;
            return result;
        }

        /// <summary>
        /// Creates an empty KV cache for this model.
        /// </summary>
        public DynamicCache CreateCache() {
            return new DynamicCache(Config.NumHiddenLayers, _numKeyValueHeads, _headDim, dataType: CacheDataType);
        }

        /// <summary>
        /// Runs <paramref name="inputIds"/> after the tokens in <paramref name="cache"/>, and adds them to the cache.
        /// </summary>
        /// <returns>The logits of the last token, which score the next token.</returns>
        public Task<float[]> ForwardAsync(IReadOnlyList<int> inputIds, DynamicCache cache) => ForwardAsync(inputIds, cache, false);

        /// <inheritdoc cref="ForwardAsync(IReadOnlyList{int}, DynamicCache)"/>
        public float[] Forward(IReadOnlyList<int> inputIds, DynamicCache cache) => SyncTask.Run(ForwardAsync(inputIds, cache, true));

        /// <summary>
        /// The most tokens of a prompt that one run of the model reads. A longer prompt runs in chunks, as vLLM's and
        /// llama.cpp's chunked prefill do, so the activations and logits of one run stay small.
        /// </summary>
        public const int PrefillChunkSize = 256;

        internal async Task<float[]> ForwardAsync(IReadOnlyList<int> ids, DynamicCache cache, bool sync) {
            var logits = Prefill(ids, cache);
            int vocab = logits.shape[-1];
            using Tensor<float> cpu = sync ? logits.ReadbackAndClone() : await logits.ReadbackAndCloneAsync();
            return cpu.AsReadOnlySpan()[^vocab..].ToArray();
        }

        /// <summary>
        /// Schedules the runs of the model on <paramref name="ids"/>, in chunks of <see cref="PrefillChunkSize"/>.
        /// </summary>
        /// <returns>The logits of the last chunk, on the device of the worker.</returns>
        private Tensor<float> Prefill(IReadOnlyList<int> ids, DynamicCache cache) {
            for (int start = 0; start + PrefillChunkSize < ids.Count; start += PrefillChunkSize) {
                Run(ids.Skip(start).Take(PrefillChunkSize).ToArray(), cache);
            }
            int last = (ids.Count - 1) / PrefillChunkSize * PrefillChunkSize;
            Run(last == 0 && ids is int[] all ? all : ids.Skip(last).ToArray(), cache);
            return (Tensor<float>)_worker.PeekOutput(_logits);
        }

        /// <summary>
        /// Schedules one run of the model on <paramref name="ids"/> after the tokens in <paramref name="cache"/>.
        /// The worker takes the keys and values of the cache and frees each after the layer that reads it, and the
        /// cache takes the new ones from the worker, so the cache is not in memory twice.
        /// </summary>
        private void Run(int[] ids, DynamicCache cache) {
            using var inputIds = new Tensor<int>(new TensorShape(1, ids.Length), ids);
            Run(inputIds, cache);
        }

        /// <inheritdoc cref="Run(int[], DynamicCache)"/>
        /// <param name="inputIds">The token ids, shape (1, tokens). They can be on the GPU, for example the token
        /// that <see cref="GpuSampler"/> picked.</param>
        private void Run(Tensor<int> inputIds, DynamicCache cache) {
            var (attentionMask, positionIds) = SetInputs(inputIds, cache);
            try {
                if (_commandBuffer != null) {
                    _commandBuffer.ScheduleWorker(_worker);
                    Graphics.ExecuteCommandBuffer(_commandBuffer);
                    _commandBuffer.Clear();
                } else {
                    _worker.Schedule();
                }
            } catch {
                cache.Clear();
                throw;
            } finally {
                attentionMask.Dispose();
                positionIds.Dispose();
            }
            TakePresent(cache);
        }

        /// <summary>
        /// Schedules one run of the model as <see cref="Run(Tensor{int}, DynamicCache)"/> does, in parts: each
        /// <c>MoveNext</c> schedules layers until <paramref name="endPart"/> returns true. The run is done, and the
        /// cache has the new keys and values, when <c>MoveNext</c> returns false.
        /// </summary>
        private System.Collections.IEnumerator RunInParts(Tensor<int> inputIds, DynamicCache cache, Func<bool> endPart) {
            var (attentionMask, positionIds) = SetInputs(inputIds, cache);
            System.Collections.IEnumerator schedule = null;
            try {
                schedule = _commandBuffer.ScheduleInParts(_worker, endPart);
                while (true) {
                    bool more;
                    try {
                        more = schedule.MoveNext();
                    } catch {
                        cache.Clear();
                        throw;
                    }
                    if (!more) {
                        break;
                    }
                    yield return null;
                }
            } finally {
                (schedule as IDisposable)?.Dispose();
                attentionMask.Dispose();
                positionIds.Dispose();
            }
            TakePresent(cache);
        }

        /// <summary>
        /// Sets the inputs of one run, and gives the keys and values of the cache to the worker.
        /// </summary>
        /// <returns>The attention mask and position ids, to dispose after the run is scheduled.</returns>
        private (Tensor<int>, Tensor<int>) SetInputs(Tensor<int> inputIds, DynamicCache cache) {
            int count = inputIds.shape[1];
            int past = cache.GetSeqLength();
            int total = past + count;
            var attentionMask = new Tensor<int>(new TensorShape(1, total), Enumerable.Repeat(1, total).ToArray());
            var positionIds = new Tensor<int>(new TensorShape(1, count), Enumerable.Range(past, count).ToArray());
            _worker.SetInput(_inputIds, inputIds);
            _worker.SetInput(_attentionMask, attentionMask);
            _worker.SetInput(_positionIds, positionIds);
            for (int i = 0; i < cache.NumLayers; i++) {
                var (key, value) = cache[i];
                bool given = _worker.TryGiveInput(_pastInputs[2 * i], key);
                given &= _worker.TryGiveInput(_pastInputs[2 * i + 1], value);
                if (given) {
                    cache.Forget(i);
                }
            }
            return (attentionMask, positionIds);
        }

        /// <summary>
        /// Takes the new keys and values of a run from the worker into the cache.
        /// </summary>
        private void TakePresent(DynamicCache cache) {
            for (int i = 0; i < cache.NumLayers; i++) {
                cache.Update(_worker.TakeOutput(_presentOutputs[2 * i]), _worker.TakeOutput(_presentOutputs[2 * i + 1]), i);
            }
        }

        public virtual void Dispose() {
            _sampler?.Dispose();
            _sampler = null;
            _worker?.Dispose();
            _commandBuffer?.Release();
        }
    }
}
