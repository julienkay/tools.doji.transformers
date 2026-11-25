using System.Collections.Generic;
using Doji.Gguf;
using Unity.Collections;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The tensors of a GGUF file as weights of a <see cref="FunctionalGraph"/>. Each weight stays in the file until
    /// the <see cref="ModelFile"/> of <see cref="Compile"/> creates a worker, and is then converted straight into the
    /// staging memory of the upload.
    /// </summary>
    /// <remarks>
    /// F32 tensors stay float32. Tensors of other types (F16, BF16, quantized) become float16, which the graph casts
    /// to float32 where a layer reads them, so their values are those of the dequantization of ggml, rounded to
    /// float16.
    /// </remarks>
    internal sealed class GgufWeights {

        private readonly ExternalWeights _weights;
        private readonly Source _source;
        private readonly Dictionary<string, FunctionalTensor> _added = new();

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
        /// The linear weight <paramref name="name"/>, [out, in] in the file, as [in, out] for <c>MatMul(x, w)</c>.
        /// </summary>
        public FunctionalTensor Linear(string name) => Add(name, true, null);

        public bool Has(string name) => File.ContainsTensor(name);

        /// <summary>
        /// Compiles the graph, as <see cref="ExternalWeights.Compile"/> does.
        /// </summary>
        public ModelFile Compile() => _weights.Compile();

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
            _source.Specs[key] = (tensor, output, transpose);
            var weight = _weights.Add(key, output == GgufOutput.Float32 ? DataType.Float : DataType.Short, new TensorShape(shape));
            // One cast for all the layers that read the weight.
            added = output == GgufOutput.Float32 ? weight : weight.Type(DataType.Float);
            _added[key] = added;
            return added;
        }

        private sealed class Source : IWeightSource {

            private readonly GgufFile _file;
            public readonly Dictionary<string, (GgufTensorInfo Tensor, GgufOutput Output, bool Transpose)> Specs = new();

            public Source(GgufFile file) => _file = file;

            public void Read(string key, NativeArray<byte> destination) {
                var (tensor, output, transpose) = Specs[key];
                _file.ReadTensor(tensor, output, transpose, destination);
            }
        }
    }
}
