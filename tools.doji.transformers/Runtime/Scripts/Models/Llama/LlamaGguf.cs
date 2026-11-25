using System;
using System.Collections.Generic;
using System.Linq;
using Doji.Gguf;
using Unity.InferenceEngine;
using F = Unity.InferenceEngine.Functional;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Decoder-only models with the layers of Llama from a GGUF file of llama.cpp: the architectures <c>llama</c>
    /// (<see cref="LlamaForCausalLM"/>, for example SmolLM2), <c>qwen3</c> (<see cref="Qwen3ForCausalLM"/>) and
    /// <c>phi3</c> (<see cref="Phi3ForCausalLM"/>). Builds their config, and their graph with the inputs and outputs of
    /// the ONNX export (<see cref="PreTrainedModel"/>).
    /// </summary>
    /// <remarks>
    /// convert_hf_to_gguf.py of llama.cpp changes the tensors of transformers, and the graph undoes that:
    /// <list type="bullet">
    /// <item>For <c>llama</c>, the rows of the query and key weights of each head are permuted: row 2j holds row j
    /// and row 2j + 1 holds row j + head_dim / 2. llama.cpp then rotates the pairs of neighbor values. The graph does
    /// the same: the rotary embedding of neighbor pairs on the permuted heads gives the dot products of the rotary
    /// embedding of transformers on the original heads, because the query and the key are permuted alike. The keys
    /// in the cache are in the permuted order, which the cache does not see.</item>
    /// <item>For <c>phi3</c>, <c>attn_qkv</c> holds the rows of the query, key and value weights, and <c>ffn_up</c>
    /// the rows of the gate and up weights, as in transformers.</item>
    /// </list>
    /// </remarks>
    internal static class LlamaGguf {

        public const string Llama = "llama", Qwen3 = "qwen3", Phi3 = "phi3";

        public static readonly string[] Architectures = { Llama, Qwen3, Phi3 };

        /// <summary>
        /// The config of the model class of transformers for the architecture of <paramref name="file"/>.
        /// </summary>
        public static PretrainedConfig Config(GgufFile file) {
            string arch = file.Architecture;
            int M(string key) => file.GetValue<int>($"{arch}.{key}");
            int hidden = M("embedding_length"), heads = M("attention.head_count");
            int window = file.GetValue($"{arch}.attention.sliding_window", 0);
            return new PretrainedConfig {
                Architectures = new List<string> { ClassName(arch) },
                ModelType = arch,
                VocabSize = checked((int)file.GetTensor("token_embd.weight").Shape[0]),
                HiddenSize = hidden,
                NumHiddenLayers = M("block_count"),
                NumAttentionHeads = heads,
                NumKeyValueHeads = file.GetValue($"{arch}.attention.head_count_kv", heads),
                HeadDim = file.GetValue($"{arch}.attention.key_length", hidden / heads),
                MaxPositionEmbeddings = M("context_length"),
                SlidingWindow = window > 0 ? window : null,
            };
        }

        private static string ClassName(string arch) => arch switch {
            Llama => "LlamaForCausalLM",
            Qwen3 => "Qwen3ForCausalLM",
            Phi3 => "Phi3ForCausalLM",
            _ => throw new NotSupportedException($"GGUF files of architecture '{arch}' are not supported here."),
        };

        /// <summary>Creates the model class of transformers for the architecture of <paramref name="file"/>.</summary>
        public static PreTrainedModel Create(GgufFile file, ModelFile model, PretrainedConfig config, GenerationConfig generationConfig, BackendType backend) {
            return file.Architecture switch {
                Llama => new LlamaForCausalLM(model, config, generationConfig, backend),
                Qwen3 => new Qwen3ForCausalLM(model, config, generationConfig, backend),
                Phi3 => new Phi3ForCausalLM(model, config, generationConfig, backend),
                _ => throw new NotSupportedException(file.Architecture),
            };
        }

        /// <summary>
        /// Builds the graph, and returns a <see cref="ModelFile"/> that reads the weights from
        /// <paramref name="file"/> when it creates a worker.
        /// </summary>
        public static ModelFile Build(GgufFile file, PretrainedConfig config) {
            string arch = file.Architecture;
            float eps = file.GetValue<float>($"{arch}.attention.layer_norm_rms_epsilon");
            float theta = file.GetValue($"{arch}.rope.freq_base", 10000f);
            int hidden = config.HiddenSize, heads = config.NumAttentionHeads, kvHeads = config.NumKeyValueHeads.Value;
            int headDim = config.HeadDim.Value, layers = config.NumHiddenLayers;
            int ropeDim = file.GetValue($"{arch}.rope.dimension_count", headDim);
            bool pairs = arch == Llama;
            if (file.ContainsTensor("rope_factors_long.weight")) {
                // The long-context Phi-3 models: transformers switches between two sets of factors by the length.
                throw new NotSupportedException($"{file.Path} has the LongRoPE factors of a long-context model, which are not supported.");
            }
            string ropeScaling = file.GetValue($"{arch}.rope.scaling.type", "none");
            if (ropeScaling is not ("none" or "")) {
                throw new NotSupportedException($"{file.Path} has the rope scaling '{ropeScaling}', which is not supported.");
            }

            var graph = new FunctionalGraph();
            var w = new GgufWeights(file, graph);

            var inputIds = graph.AddInput<int>(new DynamicTensorShape(1, -1), "input_ids");
            var attentionMask = graph.AddInput<int>(new DynamicTensorShape(1, -1), "attention_mask");
            var positionIds = graph.AddInput<int>(new DynamicTensorShape(1, -1), "position_ids");
            var past = new List<FunctionalTensor>();
            for (int i = 0; i < layers; i++) {
                past.Add(graph.AddInput<float>(new DynamicTensorShape(1, kvHeads, -1, headDim), $"past_key_values.{i}.key"));
                past.Add(graph.AddInput<float>(new DynamicTensorShape(1, kvHeads, -1, headDim), $"past_key_values.{i}.value"));
            }

            var hiddenStates = w.Embedding("token_embd.weight", inputIds.Reshape(new[] { -1 })).Reshape(new[] { 1, -1, hidden });

            // Rotary embedding of the first ropeDim values of each head. The llama3 rope scaling of Llama 3.1 and
            // later is in rope_freqs: each frequency is divided by its factor.
            var invFreq = F.Constant(new Tensor<float>(new TensorShape(ropeDim / 2),
                Enumerable.Range(0, ropeDim / 2).Select(i => 1f / MathF.Pow(theta, 2f * i / ropeDim)).ToArray()));
            if (w.Has("rope_freqs.weight")) {
                invFreq /= w.Weight("rope_freqs.weight", ropeDim / 2);
            }
            var freqs = positionIds.Float().Unsqueeze(-1) * invFreq;
            FunctionalTensor cos, sin;
            if (pairs) {
                // Each frequency for a pair of neighbor values.
                var emb = F.Concat(new[] { freqs.Unsqueeze(-1), freqs.Unsqueeze(-1) }, -1).Reshape(new[] { 1, -1, ropeDim }).Unsqueeze(1);
                (cos, sin) = (F.Cos(emb), F.Sin(emb));
            } else {
                var emb = F.Concat(new[] { freqs, freqs }, -1).Unsqueeze(1);
                (cos, sin) = (F.Cos(emb), F.Sin(emb));
            }
            FunctionalTensor Rope(FunctionalTensor x, int count) {
                var rotated = ropeDim == headDim ? x : x[.., .., .., ..ropeDim];
                FunctionalTensor turned;
                if (pairs) {
                    // (x0, x1) of each pair becomes (-x1, x0).
                    var pair = rotated.Reshape(new[] { 1, count, -1, ropeDim / 2, 2 });
                    turned = F.Concat(new[] { -pair[.., .., .., .., 1..], pair[.., .., .., .., ..1] }, -1).Reshape(new[] { 1, count, -1, ropeDim });
                } else {
                    turned = F.Concat(new[] { -rotated[.., .., .., (ropeDim / 2)..], rotated[.., .., .., ..(ropeDim / 2)] }, -1);
                }
                var result = rotated * cos + turned * sin;
                return ropeDim == headDim ? result : F.Concat(new[] { result, x[.., .., .., ropeDim..] }, -1);
            }

            // A key is visible to a query at the same or a later position, and with a sliding window, to a query less
            // than the window after it.
            var keyPositions = (F.CumSum(attentionMask, 1) - 1).Unsqueeze(1).Unsqueeze(1).Unsqueeze(1);
            var queryPositions = positionIds.Unsqueeze(-1).Unsqueeze(1).Unsqueeze(1);
            var visible = F.LessEqual(keyPositions, queryPositions);
            if (config.SlidingWindow is int window) {
                visible = F.LogicalAnd(visible, F.Greater(keyPositions, queryPositions - window));
            }
            var mask = F.Where(visible, F.Constant(0f), F.Constant(-1e9f));

            FunctionalTensor Linear(FunctionalTensor x, string name) {
                var y = w.MatMul(x, name + ".weight");
                return w.Has(name + ".bias") ? y + w.Weight(name + ".bias") : y;
            }

            var outputs = new List<(FunctionalTensor, string)>();
            int qSize = heads * headDim, kvSize = kvHeads * headDim;
            for (int i = 0; i < layers; i++) {
                string p = $"blk.{i}.";
                var residual = hiddenStates;
                var x = F.RMSNorm(hiddenStates, w.Weight(p + "attn_norm.weight"), eps);
                FunctionalTensor q, k, v;
                if (w.Has(p + "attn_qkv.weight")) {
                    var qkv = Linear(x, p + "attn_qkv");
                    (q, k, v) = (qkv[.., .., ..qSize], qkv[.., .., qSize..(qSize + kvSize)], qkv[.., .., (qSize + kvSize)..]);
                } else {
                    (q, k, v) = (Linear(x, p + "attn_q"), Linear(x, p + "attn_k"), Linear(x, p + "attn_v"));
                }
                q = q.Reshape(new[] { 1, -1, heads, headDim });
                k = k.Reshape(new[] { 1, -1, kvHeads, headDim });
                if (w.Has(p + "attn_q_norm.weight")) {
                    // Qwen3: an RMSNorm over each head of the query and the key.
                    q = F.RMSNorm(q, w.Weight(p + "attn_q_norm.weight"), eps);
                    k = F.RMSNorm(k, w.Weight(p + "attn_k_norm.weight"), eps);
                }
                q = Rope(q.Transpose(1, 2), heads);
                k = Rope(k.Transpose(1, 2), kvHeads);
                v = v.Reshape(new[] { 1, -1, kvHeads, headDim }).Transpose(1, 2);
                var keys = F.Concat(new[] { past[2 * i], k }, 2);
                var values = F.Concat(new[] { past[2 * i + 1], v }, 2);
                outputs.Add((keys, $"present.{i}.key"));
                outputs.Add((values, $"present.{i}.value"));
                // Grouped-query attention: query head h reads key/value head h / (heads / kvHeads).
                var grouped = q.Reshape(new[] { 1, kvHeads, heads / kvHeads, -1, headDim });
                var scores = F.MatMul(grouped, keys.Unsqueeze(2).Transpose(3, 4)) * (1f / MathF.Sqrt(headDim)) + mask;
                var attention = F.MatMul(F.Softmax(scores, -1), values.Unsqueeze(2)).Reshape(new[] { 1, heads, -1, headDim });
                x = Linear(attention.Transpose(1, 2).Reshape(new[] { 1, -1, qSize }), p + "attn_output");
                hiddenStates = residual + x;

                residual = hiddenStates;
                x = F.RMSNorm(hiddenStates, w.Weight(p + "ffn_norm.weight"), eps);
                FunctionalTensor gate, up;
                if (w.Has(p + "ffn_gate.weight")) {
                    (gate, up) = (w.MatMul(x, p + "ffn_gate.weight"), w.MatMul(x, p + "ffn_up.weight"));
                } else {
                    // Phi-3: the rows of the gate weight, then of the up weight.
                    var gateUp = w.MatMul(x, p + "ffn_up.weight");
                    int ff = file.GetValue<int>($"{arch}.feed_forward_length");
                    (gate, up) = (gateUp[.., .., ..ff], gateUp[.., .., ff..]);
                }
                hiddenStates = residual + w.MatMul(Silu(gate) * up, p + "ffn_down.weight");
            }
            hiddenStates = F.RMSNorm(hiddenStates, w.Weight("output_norm.weight"), eps);
            // Tied embeddings: a file without output.weight uses the embedding table as the head.
            graph.AddOutput(w.MatMul(hiddenStates, w.Has("output.weight") ? "output.weight" : "token_embd.weight"), "logits");
            foreach (var (tensor, name) in outputs) {
                graph.AddOutput(tensor, name);
            }
            return w.Compile();
        }

        private static FunctionalTensor Silu(FunctionalTensor x) => x * F.Sigmoid(x);
    }
}
