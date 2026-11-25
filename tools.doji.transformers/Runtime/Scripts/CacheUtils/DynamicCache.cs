using System;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A cache that grows with each token. It owns its tensors, which stay on the model's device.
    /// </summary>
    /// <remarks>
    /// In a model with linear attention layers (Qwen3.5), the cache of such a layer is two states that each step
    /// replaces instead of grows, as transformers' <c>LinearAttentionLayer</c>: the last inputs of the causal
    /// convolution, [batch, channels, kernel - 1], and the state of the gated delta rule, [batch, value heads,
    /// key size, value size]. Both are float32 and start as zeros.
    /// </remarks>
    public class DynamicCache : Cache {

        private readonly Tensor[] _keys;
        private readonly Tensor[] _values;
        private readonly TensorShape _empty;
        private readonly bool[] _linear;
        private readonly TensorShape _convShape, _recurrentShape;
        private readonly int _firstAttentionLayer;

        /// <summary>
        /// The data type of the keys and values: <see cref="DataType.Float"/>, or <see cref="DataType.Short"/> for
        /// float16.
        /// </summary>
        public DataType DataType { get; }

        /// <summary>
        /// Creates an empty cache, with zero tokens per layer.
        /// </summary>
        public DynamicCache(int numLayers, int numKeyValueHeads, int headDim, int batchSize = 1, DataType dataType = DataType.Float)
            : this(numLayers, numKeyValueHeads, headDim, batchSize, dataType, new bool[numLayers], default, default) { }

        /// <summary>
        /// Creates an empty cache for a model, as transformers' <c>DynamicCache(config=config)</c>: with the states of
        /// linear attention layers where <see cref="PretrainedConfig.LayerTypes"/> has them.
        /// </summary>
        public DynamicCache(PretrainedConfig config, int batchSize = 1, DataType dataType = DataType.Float)
            : this(config.GetTextConfig().NumHiddenLayers, KeyValueHeads(config.GetTextConfig()), HeadDim(config.GetTextConfig()),
                batchSize, dataType, LinearLayers(config.GetTextConfig()), ConvShape(config.GetTextConfig(), batchSize),
                RecurrentShape(config.GetTextConfig(), batchSize)) { }

        internal static int KeyValueHeads(PretrainedConfig text) => text.NumKeyValueHeads ?? text.NumAttentionHeads;

        internal static int HeadDim(PretrainedConfig text) => text.HeadDim ?? text.HiddenSize / text.NumAttentionHeads;

        private DynamicCache(int numLayers, int numKeyValueHeads, int headDim, int batchSize, DataType dataType, bool[] linear,
            TensorShape convShape, TensorShape recurrentShape) {
            if (dataType != DataType.Float && dataType != DataType.Short) {
                throw new ArgumentException("The cache holds Float or Short (float16) tensors.", nameof(dataType));
            }
            DataType = dataType;
            _keys = new Tensor[numLayers];
            _values = new Tensor[numLayers];
            _empty = new TensorShape(batchSize, numKeyValueHeads, 0, headDim);
            _linear = linear;
            _convShape = convShape;
            _recurrentShape = recurrentShape;
            _firstAttentionLayer = Array.IndexOf(linear, false);
            for (int i = 0; i < numLayers; i++) {
                (_keys[i], _values[i]) = Empty(i);
            }
        }

        private static bool[] LinearLayers(PretrainedConfig text) {
            var linear = new bool[text.NumHiddenLayers];
            for (int i = 0; i < linear.Length; i++) {
                linear[i] = text.IsLinearAttention(i);
            }
            return linear;
        }

        private static TensorShape ConvShape(PretrainedConfig text, int batchSize) {
            if (text.LayerTypes == null || !text.LayerTypes.Contains("linear_attention")) {
                return default;
            }
            int keyDim = text.LinearNumKeyHeads.Value * text.LinearKeyHeadDim.Value;
            int valueDim = text.LinearNumValueHeads.Value * text.LinearValueHeadDim.Value;
            return new TensorShape(batchSize, 2 * keyDim + valueDim, text.LinearConvKernelDim.Value - 1);
        }

        private static TensorShape RecurrentShape(PretrainedConfig text, int batchSize) {
            if (text.LayerTypes == null || !text.LayerTypes.Contains("linear_attention")) {
                return default;
            }
            return new TensorShape(batchSize, text.LinearNumValueHeads.Value, text.LinearKeyHeadDim.Value, text.LinearValueHeadDim.Value);
        }

        private (Tensor, Tensor) Empty(int layerIdx) {
            if (_linear[layerIdx]) {
                return (new Tensor<float>(_convShape), new Tensor<float>(_recurrentShape));
            }
            return DataType == DataType.Short
                ? (new Tensor<short>(_empty), new Tensor<short>(_empty))
                : (new Tensor<float>(_empty), new Tensor<float>(_empty));
        }

        public int NumLayers => _keys.Length;

        /// <summary>
        /// Whether layer <paramref name="layerIdx"/> is a linear attention layer, whose cache is two states.
        /// </summary>
        public bool IsLinearAttention(int layerIdx) => _linear[layerIdx];

        /// <summary>
        /// The cached keys and values of a layer, or the conv and recurrent states of a linear attention layer.
        /// </summary>
        public override (Tensor Key, Tensor Value) this[int layerIdx] => (_keys[layerIdx], _values[layerIdx]);

        /// <summary>
        /// The number of cached tokens. A linear attention layer does not count tokens: as in transformers, the
        /// first full attention layer gives the count.
        /// </summary>
        public override int GetSeqLength(int layerIdx = 0) {
            if (_linear[layerIdx]) {
                if (layerIdx != 0 || _firstAttentionLayer < 0) {
                    throw new ArgumentException($"Layer {layerIdx} is a linear attention layer, which does not count tokens.", nameof(layerIdx));
                }
                layerIdx = _firstAttentionLayer;
            }
            return _keys[layerIdx].shape[2];
        }

        /// <summary>
        /// Replaces the keys and values of a layer with ones that include the new tokens, or the states of a linear
        /// attention layer with the next ones, and disposes the old ones.
        /// </summary>
        public void Update(Tensor keyStates, Tensor valueStates, int layerIdx) {
            _keys[layerIdx]?.Dispose();
            _values[layerIdx]?.Dispose();
            _keys[layerIdx] = keyStates;
            _values[layerIdx] = valueStates;
        }

        /// <summary>
        /// Drops the keys and values of a layer without disposing them, after the model took them.
        /// </summary>
        internal void Forget(int layerIdx) {
            _keys[layerIdx] = null;
            _values[layerIdx] = null;
        }

        /// <summary>
        /// Disposes all keys, values and states, and empties the cache.
        /// </summary>
        public void Clear() {
            for (int i = 0; i < _keys.Length; i++) {
                _keys[i]?.Dispose();
                _values[i]?.Dispose();
                (_keys[i], _values[i]) = Empty(i);
            }
        }

        public override void Dispose() {
            for (int i = 0; i < _keys.Length; i++) {
                _keys[i]?.Dispose();
                _values[i]?.Dispose();
            }
        }
    }
}
