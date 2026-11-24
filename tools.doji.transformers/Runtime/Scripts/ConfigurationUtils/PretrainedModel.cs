using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.InferenceEngine;

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
        }

        /// <summary>
        /// Creates the model from an open .sentis file. The weights go from the file to the worker, so they are
        /// never in managed memory. The file can be disposed afterwards.
        /// </summary>
        protected PreTrainedModel(ModelFile file, PretrainedConfig config, GenerationConfig generationConfig, BackendType backend)
            : this(file?.Inputs, file?.Outputs, config, generationConfig) {
            _worker = file.CreateWorker(backend);
        }

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
            return new DynamicCache(Config.NumHiddenLayers, _numKeyValueHeads, _headDim);
        }

        /// <summary>
        /// Runs <paramref name="inputIds"/> after the tokens in <paramref name="cache"/>, and adds them to the cache.
        /// </summary>
        /// <returns>The logits of the last token, which score the next token.</returns>
        public Task<float[]> ForwardAsync(IReadOnlyList<int> inputIds, DynamicCache cache) => ForwardAsync(inputIds, cache, false);

        /// <inheritdoc cref="ForwardAsync(IReadOnlyList{int}, DynamicCache)"/>
        public float[] Forward(IReadOnlyList<int> inputIds, DynamicCache cache) => SyncTask.Run(ForwardAsync(inputIds, cache, true));

        internal async Task<float[]> ForwardAsync(IReadOnlyList<int> ids, DynamicCache cache, bool sync) {
            int past = cache.GetSeqLength();
            int total = past + ids.Count;
            using var inputIds = new Tensor<int>(new TensorShape(1, ids.Count), ids.ToArray());
            using var attentionMask = new Tensor<int>(new TensorShape(1, total), Enumerable.Repeat(1, total).ToArray());
            using var positionIds = new Tensor<int>(new TensorShape(1, ids.Count), Enumerable.Range(past, ids.Count).ToArray());
            _worker.SetInput(_inputIds, inputIds);
            _worker.SetInput(_attentionMask, attentionMask);
            _worker.SetInput(_positionIds, positionIds);
            for (int i = 0; i < cache.NumLayers; i++) {
                _worker.SetInput(_pastInputs[2 * i], cache[i].Key);
                _worker.SetInput(_pastInputs[2 * i + 1], cache[i].Value);
            }
            _worker.Schedule();

            // The worker reuses its outputs on the next run, so the cache keeps copies.
            for (int i = 0; i < cache.NumLayers; i++) {
                Tensor key = null, value = null;
                _worker.CopyOutput(_presentOutputs[2 * i], ref key);
                _worker.CopyOutput(_presentOutputs[2 * i + 1], ref value);
                cache.Update((Tensor<float>)key, (Tensor<float>)value, i);
            }

            var logits = (Tensor<float>)_worker.PeekOutput(_logits);
            int vocab = logits.shape[-1];
            using Tensor<float> cpu = sync ? logits.ReadbackAndClone() : await logits.ReadbackAndCloneAsync();
            return cpu.AsReadOnlySpan()[^vocab..].ToArray();
        }

        public virtual void Dispose() {
            _worker?.Dispose();
        }
    }
}
