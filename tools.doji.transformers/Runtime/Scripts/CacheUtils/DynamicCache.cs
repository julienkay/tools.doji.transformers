using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A cache that grows with each token. It owns its tensors, which stay on the model's device.
    /// </summary>
    public class DynamicCache : Cache {

        private readonly Tensor[] _keys;
        private readonly Tensor[] _values;
        private readonly TensorShape _empty;

        /// <summary>
        /// The data type of the keys and values: <see cref="DataType.Float"/>, or <see cref="DataType.Short"/> for
        /// float16.
        /// </summary>
        public DataType DataType { get; }

        /// <summary>
        /// Creates an empty cache, with zero tokens per layer.
        /// </summary>
        public DynamicCache(int numLayers, int numKeyValueHeads, int headDim, int batchSize = 1, DataType dataType = DataType.Float) {
            if (dataType != DataType.Float && dataType != DataType.Short) {
                throw new System.ArgumentException("The cache holds Float or Short (float16) tensors.", nameof(dataType));
            }
            DataType = dataType;
            _keys = new Tensor[numLayers];
            _values = new Tensor[numLayers];
            _empty = new TensorShape(batchSize, numKeyValueHeads, 0, headDim);
            for (int i = 0; i < numLayers; i++) {
                _keys[i] = Empty();
                _values[i] = Empty();
            }
        }

        private Tensor Empty() => DataType == DataType.Short ? new Tensor<short>(_empty) : new Tensor<float>(_empty);

        public int NumLayers => _keys.Length;

        public override (Tensor Key, Tensor Value) this[int layerIdx] => (_keys[layerIdx], _values[layerIdx]);

        public override int GetSeqLength(int layerIdx = 0) => _keys[layerIdx].shape[2];

        /// <summary>
        /// Replaces the keys and values of a layer with ones that include the new tokens, and disposes the old ones.
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
        /// Disposes all keys and values and empties the cache.
        /// </summary>
        public void Clear() {
            for (int i = 0; i < _keys.Length; i++) {
                _keys[i]?.Dispose();
                _values[i]?.Dispose();
                _keys[i] = Empty();
                _values[i] = Empty();
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
