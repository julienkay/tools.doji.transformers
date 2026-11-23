using System;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The keys and values of the attention layers for the tokens seen so far, so that each step of generation only
    /// runs the new tokens.
    /// </summary>
    public abstract class Cache : IDisposable {

        /// <summary>
        /// The number of cached tokens.
        /// </summary>
        public abstract int GetSeqLength(int layerIdx = 0);

        /// <summary>
        /// The cached keys and values of a layer, shaped [batch, kv_heads, tokens, head_dim].
        /// </summary>
        public abstract (Tensor<float> Key, Tensor<float> Value) this[int layerIdx] { get; }

        public abstract void Dispose();
    }
}
