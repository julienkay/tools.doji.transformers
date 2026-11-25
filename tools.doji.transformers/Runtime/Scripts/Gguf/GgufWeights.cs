using System.Collections.Generic;
using Doji.Gguf;
using Unity.Collections;
using Unity.InferenceEngine;
using F = Unity.InferenceEngine.Functional;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The tensors of a GGUF file as weights of a <see cref="FunctionalGraph"/>. Each weight stays in the file until
    /// the <see cref="ModelFile"/> of <see cref="Compile"/> creates a worker, and is then converted straight into the
    /// staging memory of the upload.
    /// </summary>
    /// <remarks>
    /// <para>Linear and embedding weights of type Q8_0, Q4_K, Q5_K or Q6_K stay in their blocks on the device
    /// (<see cref="BlockQuant"/>): <see cref="MatMul"/> and <see cref="Embedding"/> read them with the layers of
    /// BlockQuant, which compute with the values of the dequantization of ggml.</para>
    /// <para>All other weights are converted. F32 tensors stay float32. Tensors of other types (F16, BF16, other
    /// quantized types) become float16, which the graph casts to float32 where a layer reads them, so their values are
    /// those of the dequantization of ggml, rounded to float16.</para>
    /// </remarks>
    internal sealed class GgufWeights {

        private readonly ExternalWeights _weights;
        private readonly Source _source;
        private readonly Dictionary<string, FunctionalTensor> _added = new();
        private readonly SortedSet<string> _converted = new();

        public GgufFile File { get; }

        public GgufWeights(GgufFile file, FunctionalGraph graph) {
            File = file;
            _source = new Source(file);
            _weights = new ExternalWeights(graph, _source);
        }

        /// <summary>
        /// The tensor <paramref name="name"/> in its row-major shape, or reshaped to <paramref name="shape"/>.
        /// </summary>
        public FunctionalTensor Weight(string name, params int[] shape) => Add(name, false, shape);

        /// <summary>
        /// <c>x @ Wᵀ</c> for the linear weight <paramref name="name"/> (out rows of in values in the file).
        /// </summary>
        public FunctionalTensor MatMul(FunctionalTensor x, string name) {
            var tensor = File.GetTensor(name);
            if (TryPacked(tensor, out var weight, out var format)) {
                return BlockQuant.MatMul(x, weight, format, (int)tensor.Columns, (int)tensor.Rows);
            }
            return F.MatMul(x, Add(name, true, null));
        }

        /// <summary>The rows <paramref name="ids"/> of the embedding table <paramref name="name"/>.</summary>
        public FunctionalTensor Embedding(string name, FunctionalTensor ids) {
            var tensor = File.GetTensor(name);
            if (TryPacked(tensor, out var weight, out var format)) {
                return BlockQuant.Gather(weight, ids, format, (int)tensor.Columns, (int)tensor.Rows);
            }
            return Add(name, false, null).IndexSelect(0, ids);
        }

        public bool Has(string name) => File.ContainsTensor(name);

        /// <summary>
        /// Compiles the graph, as <see cref="ExternalWeights.Compile"/> does. Logs a warning for quantized tensors
        /// that became float16.
        /// </summary>
        public ModelFile Compile() {
            if (_converted.Count > 0) {
                UnityEngine.Debug.LogWarning($"{File.Path}: tensors of type {string.Join(", ", _converted)} load as float16, which takes more memory. " +
                    "Q8_0, Q4_K, Q5_K and Q6_K tensors stay quantized on the device.");
            }
            return _weights.Compile();
        }

        private static bool TryFormat(GgufType type, out BlockQuantFormat format) {
            switch (type) {
                case GgufType.Q8_0: format = BlockQuantFormat.Q8_0; return true;
                case GgufType.Q4_K: format = BlockQuantFormat.Q4_K; return true;
                case GgufType.Q6_K: format = BlockQuantFormat.Q6_K; return true;
                case GgufType.Q5_K: format = BlockQuantFormat.Q5_K; return true;
                default: format = default; return false;
            }
        }

        private bool TryPacked(GgufTensorInfo tensor, out FunctionalTensor weight, out BlockQuantFormat format) {
            weight = null;
            if (tensor.Dimensions.Length != 2 || !TryFormat(tensor.Type, out format) || !BlockQuant.Fits(format, (int)tensor.Columns)) {
                format = default;
                return false;
            }
            string key = $"{tensor.Name}|{format}";
            if (!_added.TryGetValue(key, out weight)) {
                _source.Specs[key] = new Spec { Tensor = tensor, Packed = true, Format = format };
                weight = _weights.Add(key, DataType.Int, BlockQuant.PackedShape(format, (int)tensor.Columns, (int)tensor.Rows));
                _added[key] = weight;
            }
            return true;
        }

        private FunctionalTensor Add(string name, bool transpose, int[] shape) {
            var tensor = File.GetTensor(name);
            if (!GgufConvert.IsSupported(tensor.Type)) {
                throw new System.NotSupportedException($"Tensor {tensor} of {File.Path} has a type that tools.doji.transformers does not convert.");
            }
            if (shape == null || shape.Length == 0) {
                long[] source = tensor.Shape;
                if (transpose) {
                    shape = new[] { (int)tensor.Columns, (int)tensor.Rows };
                } else {
                    shape = new int[source.Length];
                    for (int i = 0; i < source.Length; i++) {
                        shape[i] = checked((int)source[i]);
                    }
                }
            }
            var output = tensor.Type == GgufType.F32 ? GgufOutput.Float32 : GgufOutput.Float16;
            string key = $"{name}{(transpose ? "|T" : "")}|{string.Join(",", shape)}";
            if (_added.TryGetValue(key, out var added)) {
                return added;
            }
            if (tensor.Type != GgufType.F32 && tensor.Type != GgufType.F16 && tensor.Type != GgufType.BF16) {
                _converted.Add(tensor.Type.ToString());
            }
            _source.Specs[key] = new Spec { Tensor = tensor, Output = output, Transpose = transpose };
            var weight = _weights.Add(key, output == GgufOutput.Float32 ? DataType.Float : DataType.Short, new TensorShape(shape));
            // One cast for all the layers that read the weight.
            added = output == GgufOutput.Float32 ? weight : weight.Type(DataType.Float);
            _added[key] = added;
            return added;
        }

        private sealed class Spec {
            public GgufTensorInfo Tensor;
            public GgufOutput Output;
            public bool Transpose;
            public bool Packed;
            public BlockQuantFormat Format;
        }

        private sealed class Source : IWeightSource {

            private readonly GgufFile _file;
            public readonly Dictionary<string, Spec> Specs = new();

            public Source(GgufFile file) => _file = file;

            public void Read(string key, NativeArray<byte> destination) {
                var spec = Specs[key];
                if (!spec.Packed) {
                    _file.ReadTensor(spec.Tensor, spec.Output, spec.Transpose, destination);
                    return;
                }
                using var blocks = new NativeArray<byte>(checked((int)spec.Tensor.ByteCount), Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                _file.ReadTensor(spec.Tensor, blocks);
                BlockQuant.Pack(spec.Format, blocks, (int)spec.Tensor.Columns, (int)spec.Tensor.Rows, destination);
            }
        }
    }
}
