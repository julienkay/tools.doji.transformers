using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A cache that grows with each token. It owns its tensors, which stay on the model's device.
    /// </summary>
    public class DynamicCache : Cache {

        private readonly Tensor<float>[] _keys;
        private readonly Tensor<float>[] _values;
        private readonly TensorShape _empty;

        /// <summary>
        /// Creates an empty cache, with zero tokens per layer.
        /// </summary>
        public DynamicCache(int numLayers, int numKeyValueHeads, int headDim, int batchSize = 1) {
            _keys = new Tensor<float>[numLayers];
            _values = new Tensor<float>[numLayers];
            var empty = _empty = new TensorShape(batchSize, numKeyValueHeads, 0, headDim);
            for (int i = 0; i < numLayers; i++) {
                _keys[i] = new Tensor<float>(empty);
                _values[i] = new Tensor<float>(empty);
            }
        }

        public int NumLayers => _keys.Length;

        public override (Tensor<float> Key, Tensor<float> Value) this[int layerIdx] => (_keys[layerIdx], _values[layerIdx]);

        public override int GetSeqLength(int layerIdx = 0) => _keys[layerIdx].shape[2];

        /// <summary>
        /// Replaces the keys and values of a layer with ones that include the new tokens, and disposes the old ones.
        /// </summary>
        public void Update(Tensor<float> keyStates, Tensor<float> valueStates, int layerIdx) {
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
                _keys[i] = new Tensor<float>(_empty);
                _values[i] = new Tensor<float>(_empty);
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
