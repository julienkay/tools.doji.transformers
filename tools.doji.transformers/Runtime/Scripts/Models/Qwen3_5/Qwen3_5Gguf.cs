using System;
using System.Collections.Generic;
using System.Linq;
using Doji.Gguf;
using Unity.InferenceEngine;
using F = Unity.InferenceEngine.Functional;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Qwen3.5 from a GGUF file of llama.cpp (architecture <c>qwen35</c>): its config, and its graph with the inputs
    /// and outputs of the ONNX export (<see cref="PreTrainedModel"/>).
    /// </summary>
    /// <remarks>
    /// convert_hf_to_gguf.py of llama.cpp changes the tensors of transformers, and the graph undoes that:
    /// <list type="bullet">
    /// <item>The norm weights of <c>Qwen3_5RMSNorm</c> hold 1 + weight. The gated norm of the linear attention
    /// (<c>ssm_norm</c>) holds its weight.</item>
    /// <item><c>ssm_a</c> holds -exp(A_log).</item>
    /// <item>With fewer key heads than value heads, the value heads are in tiled order (value head j uses key head
    /// j % key heads), not grouped by key head as in transformers. The graph repeats the query and key heads tiled
    /// to match. The recurrent state is in the same order, which the cache does not see.</item>
    /// <item>A newer converter merges <c>attn_qkv</c> and <c>attn_gate</c> into <c>attn_qkv_gate</c>, and
    /// <c>ssm_beta</c> and <c>ssm_alpha</c> into <c>ssm_beta_alpha</c> (the rows of the first, then of the second).
    /// The graph reads either form.</item>
    /// </list>
    /// </remarks>
    internal static class Qwen3_5Gguf {

        public const string Architecture = "qwen35";

        /// <summary>
        /// The config of transformers' Qwen3_5ForCausalLM, from the metadata of <paramref name="file"/>.
        /// </summary>
        public static PretrainedConfig Config(GgufFile file) {
            int M(string key) => file.GetValue<int>($"{Architecture}.{key}");
            int layers = M("block_count") - file.GetValue($"{Architecture}.nextn_predict_layers", 0);
            int interval = M("full_attention_interval");
            int valueHeads = M("ssm.time_step_rank");
            return new PretrainedConfig {
                Architectures = new List<string> { "Qwen3_5ForCausalLM" },
                ModelType = "qwen3_5_text",
                VocabSize = checked((int)file.GetTensor("token_embd.weight").Shape[0]),
                HiddenSize = M("embedding_length"),
                NumHiddenLayers = layers,
                NumAttentionHeads = M("attention.head_count"),
                NumKeyValueHeads = M("attention.head_count_kv"),
                HeadDim = M("attention.key_length"),
                MaxPositionEmbeddings = M("context_length"),
                // As transformers' Qwen3_5TextConfig: every full_attention_interval-th layer has full attention.
                LayerTypes = Enumerable.Range(0, layers).Select(i => (i + 1) % interval == 0 ? "full_attention" : "linear_attention").ToList(),
                LinearConvKernelDim = M("ssm.conv_kernel"),
                LinearKeyHeadDim = M("ssm.state_size"),
                LinearValueHeadDim = M("ssm.inner_size") / valueHeads,
                LinearNumKeyHeads = M("ssm.group_count"),
                LinearNumValueHeads = valueHeads,
            };
        }

        /// <summary>
        /// Builds the graph, and returns a <see cref="ModelFile"/> that reads the weights from
        /// <paramref name="file"/> when it creates a worker.
        /// </summary>
        public static ModelFile Build(GgufFile file, PretrainedConfig config) {
            int M(string key) => file.GetValue<int>($"{Architecture}.{key}");
            float eps = file.GetValue<float>($"{Architecture}.attention.layer_norm_rms_epsilon");
            float theta = file.GetValue<float>($"{Architecture}.rope.freq_base");
            int hidden = config.HiddenSize, heads = config.NumAttentionHeads, kvHeads = config.NumKeyValueHeads.Value;
            int headDim = config.HeadDim.Value, ropeDim = M("rope.dimension_count");
            int kHeads = config.LinearNumKeyHeads.Value, vHeads = config.LinearNumValueHeads.Value;
            int kDim = config.LinearKeyHeadDim.Value, vDim = config.LinearValueHeadDim.Value, kernel = config.LinearConvKernelDim.Value;
            int keyDim = kHeads * kDim, valueDim = vHeads * vDim, convDim = 2 * keyDim + valueDim;
            int layers = config.NumHiddenLayers;
            if (vHeads % kHeads != 0) {
                throw new NotSupportedException($"{vHeads} value heads are not a multiple of {kHeads} key heads.");
            }

            var graph = new FunctionalGraph();
            var w = new GgufWeights(file, graph);

            var inputIds = graph.AddInput<int>(new DynamicTensorShape(1, -1), "input_ids");
            var attentionMask = graph.AddInput<int>(new DynamicTensorShape(1, -1), "attention_mask");
            var positionIds = graph.AddInput<int>(new DynamicTensorShape(1, -1), "position_ids");
            var past = new List<FunctionalTensor>();
            for (int i = 0; i < layers; i++) {
                if (config.IsLinearAttention(i)) {
                    past.Add(graph.AddInput<float>(new DynamicTensorShape(1, convDim, kernel - 1), $"past_conv.{i}"));
                    past.Add(graph.AddInput<float>(new DynamicTensorShape(1, vHeads, kDim, vDim), $"past_recurrent.{i}"));
                } else {
                    past.Add(graph.AddInput<float>(new DynamicTensorShape(1, kvHeads, -1, headDim), $"past_key_values.{i}.key"));
                    past.Add(graph.AddInput<float>(new DynamicTensorShape(1, kvHeads, -1, headDim), $"past_key_values.{i}.value"));
                }
            }

            var hiddenStates = w.Weight("token_embd.weight").IndexSelect(0, inputIds.Reshape(new[] { -1 })).Reshape(new[] { 1, -1, hidden });

            // Rotary embedding of the first ropeDim values of each head, rotate_half style. Qwen3.5 uses interleaved
            // MRoPE, which for text (the same position in all three sections) is the plain rotary embedding.
            var invFreq = Enumerable.Range(0, ropeDim / 2).Select(i => 1f / MathF.Pow(theta, 2f * i / ropeDim)).ToArray();
            var freqs = positionIds.Float().Unsqueeze(-1) * F.Constant(new Tensor<float>(new TensorShape(ropeDim / 2), invFreq));
            var emb = F.Concat(new[] { freqs, freqs }, -1).Unsqueeze(1);
            FunctionalTensor cos = F.Cos(emb), sin = F.Sin(emb);
            FunctionalTensor Rope(FunctionalTensor x) {
                var rotated = x[.., .., .., ..ropeDim];
                var half = F.Concat(new[] { -rotated[.., .., .., (ropeDim / 2)..], rotated[.., .., .., ..(ropeDim / 2)] }, -1);
                var result = rotated * cos + half * sin;
                return ropeDim == headDim ? result : F.Concat(new[] { result, x[.., .., .., ropeDim..] }, -1);
            }

            // A key is visible to a query at the same or a later position.
            var keyPositions = F.CumSum(attentionMask, 1) - 1;
            var visible = F.LessEqual(keyPositions.Unsqueeze(1).Unsqueeze(1).Unsqueeze(1), positionIds.Unsqueeze(-1).Unsqueeze(1).Unsqueeze(1));
            var mask = F.Where(visible, F.Constant(0f), F.Constant(-1e9f));

            var outputs = new List<(FunctionalTensor, string)>();
            for (int i = 0; i < layers; i++) {
                string p = $"blk.{i}.";
                var residual = hiddenStates;
                var x = F.RMSNorm(hiddenStates, w.Weight(p + "attn_norm.weight"), eps);
                if (config.IsLinearAttention(i)) {
                    FunctionalTensor mixed, z;
                    if (w.Has(p + "attn_qkv_gate.weight")) {
                        var qkvz = F.MatMul(x, w.Linear(p + "attn_qkv_gate.weight"));
                        mixed = qkvz[.., .., ..convDim];
                        z = qkvz[.., .., convDim..];
                    } else {
                        mixed = F.MatMul(x, w.Linear(p + "attn_qkv.weight"));
                        z = F.MatMul(x, w.Linear(p + "attn_gate.weight"));
                    }
                    FunctionalTensor b, a;
                    if (w.Has(p + "ssm_beta_alpha.weight")) {
                        var ba = F.MatMul(x, w.Linear(p + "ssm_beta_alpha.weight"));
                        b = ba[.., .., ..vHeads];
                        a = ba[.., .., vHeads..];
                    } else {
                        b = F.MatMul(x, w.Linear(p + "ssm_beta.weight"));
                        a = F.MatMul(x, w.Linear(p + "ssm_alpha.weight"));
                    }

                    // Causal depthwise convolution over the cached and the new tokens.
                    var convIn = F.Concat(new[] { past[2 * i], mixed.Transpose(1, 2) }, 2);
                    outputs.Add((convIn[.., .., ^(kernel - 1)..], $"present_conv.{i}"));
                    var conv = Silu(F.Conv1D(convIn, w.Weight(p + "ssm_conv1d.weight", convDim, 1, kernel), null, groups: convDim)).Transpose(1, 2);
                    var q = L2Norm(conv[.., .., ..keyDim].Reshape(new[] { 1, -1, kHeads, kDim }));
                    var k = L2Norm(conv[.., .., keyDim..(2 * keyDim)].Reshape(new[] { 1, -1, kHeads, kDim }));
                    var v = conv[.., .., (2 * keyDim)..].Reshape(new[] { 1, -1, vHeads, vDim });
                    if (vHeads != kHeads) {
                        q = F.Concat(Enumerable.Repeat(q, vHeads / kHeads).ToArray(), 2);
                        k = F.Concat(Enumerable.Repeat(k, vHeads / kHeads).ToArray(), 2);
                    }
                    var beta = F.Sigmoid(b);
                    var g = w.Weight(p + "ssm_a") * F.Softplus(a + w.Weight(p + "ssm_dt.bias"));
                    var (o, state) = LinearAttention.GatedDeltaRule(q, k, v, g, beta, past[2 * i + 1], 1f / MathF.Sqrt(kDim));
                    outputs.Add((state, $"present_recurrent.{i}"));
                    var gated = F.RMSNorm(o, w.Weight(p + "ssm_norm.weight"), eps) * Silu(z.Reshape(new[] { 1, -1, vHeads, vDim }));
                    x = F.MatMul(gated.Reshape(new[] { 1, -1, valueDim }), w.Linear(p + "ssm_out.weight"));
                } else {
                    // The query projection gives the query and an output gate for each head.
                    var qg = F.MatMul(x, w.Linear(p + "attn_q.weight")).Reshape(new[] { 1, -1, heads, 2 * headDim });
                    var gate = qg[.., .., .., headDim..].Reshape(new[] { 1, -1, heads * headDim });
                    var q = Rope(F.RMSNorm(qg[.., .., .., ..headDim], w.Weight(p + "attn_q_norm.weight"), eps).Transpose(1, 2));
                    var k = F.MatMul(x, w.Linear(p + "attn_k.weight")).Reshape(new[] { 1, -1, kvHeads, headDim });
                    k = Rope(F.RMSNorm(k, w.Weight(p + "attn_k_norm.weight"), eps).Transpose(1, 2));
                    var v = F.MatMul(x, w.Linear(p + "attn_v.weight")).Reshape(new[] { 1, -1, kvHeads, headDim }).Transpose(1, 2);
                    var keys = F.Concat(new[] { past[2 * i], k }, 2);
                    var values = F.Concat(new[] { past[2 * i + 1], v }, 2);
                    outputs.Add((keys, $"present.{i}.key"));
                    outputs.Add((values, $"present.{i}.value"));
                    // Grouped-query attention: query head h reads key/value head h / (heads / kvHeads).
                    var grouped = q.Reshape(new[] { 1, kvHeads, heads / kvHeads, -1, headDim });
                    var scores = F.MatMul(grouped, keys.Unsqueeze(2).Transpose(3, 4)) * (1f / MathF.Sqrt(headDim)) + mask;
                    var attention = F.MatMul(F.Softmax(scores, -1), values.Unsqueeze(2)).Reshape(new[] { 1, heads, -1, headDim });
                    attention = attention.Transpose(1, 2).Reshape(new[] { 1, -1, heads * headDim }) * F.Sigmoid(gate);
                    x = F.MatMul(attention, w.Linear(p + "attn_output.weight"));
                }
                hiddenStates = residual + x;
                residual = hiddenStates;
                x = F.RMSNorm(hiddenStates, w.Weight(p + "post_attention_norm.weight"), eps);
                x = F.MatMul(Silu(F.MatMul(x, w.Linear(p + "ffn_gate.weight"))) * F.MatMul(x, w.Linear(p + "ffn_up.weight")),
                    w.Linear(p + "ffn_down.weight"));
                hiddenStates = residual + x;
            }
            hiddenStates = F.RMSNorm(hiddenStates, w.Weight("output_norm.weight"), eps);
            // Tied embeddings: a file without output.weight uses the embedding table as the head.
            var head = w.Linear(w.Has("output.weight") ? "output.weight" : "token_embd.weight");
            graph.AddOutput(F.MatMul(hiddenStates, head), "logits");
            foreach (var (tensor, name) in outputs) {
                graph.AddOutput(tensor, name);
            }
            return w.Compile();
        }

        private static FunctionalTensor Silu(FunctionalTensor x) => x * F.Sigmoid(x);

        /// <summary>As transformers' <c>l2norm</c> with eps 1e-6.</summary>
        private static FunctionalTensor L2Norm(FunctionalTensor x) => x / F.Sqrt(F.ReduceSum(x * x, -1, true) + 1e-6f);
    }
}
