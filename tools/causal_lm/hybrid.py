"""
The ONNX export of Qwen3.5, whose layers mix full attention with the linear attention of Gated DeltaNet.

One model runs every step, like the export of make_models.py. A full attention layer i has the KV cache of the
other models. A linear attention layer i has two states that each step replaces:

  past_conv.{i}       [batch, conv channels, conv kernel - 1]  the last inputs of the causal convolution
  past_recurrent.{i}  [batch, value heads, key size, value size] the state of the gated delta rule
  present_conv.{i}, present_recurrent.{i}: the same after the step

Both start as zeros. The gated delta rule is a node GatedDeltaRule of the domain ai.doji, which
tools.doji.sentis-utils runs in Unity. onnxruntime cannot run it, so the checks compare this module in torch
with the model of transformers.
"""
import torch
import torch.nn.functional as F

DOMAIN = "ai.doji"


def is_hybrid(config):
    return getattr(config, "model_type", None) in ("qwen3_5_text", "qwen3_5")


def text_config(config):
    return getattr(config, "text_config", None) or config


class GatedDeltaRule(torch.autograd.Function):
    """transformers' torch_recurrent_gated_delta_rule without the l2norm, exported as one ai.doji node."""

    @staticmethod
    def forward(ctx, query, key, value, g, beta, state, scale):
        state = state.clone()
        outputs = []
        for t in range(query.shape[1]):
            state = state * g[:, t].exp()[..., None, None]
            k_t = key[:, t]
            memory = (state * k_t[..., None]).sum(dim=-2)
            delta = (value[:, t] - memory) * beta[:, t][..., None]
            state = state + k_t[..., None] * delta[..., None, :]
            outputs.append((state * (query[:, t] * scale)[..., None]).sum(dim=-2))
        output = torch.stack(outputs, dim=1) if outputs else value.clone()
        return output, state

    @staticmethod
    def symbolic(g, query, key, value, decay, beta, state, scale):
        return g.op(f"{DOMAIN}::GatedDeltaRule", query, key, value, decay, beta, state, scale_f=scale, outputs=2)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def linear_attention(attn, hidden_states, conv_state, recurrent_state):
    """Qwen3_5GatedDeltaNet.forward with the states as inputs and outputs."""
    batch, seq, _ = hidden_states.shape
    mixed = attn.in_proj_qkv(hidden_states).transpose(1, 2)
    z = attn.in_proj_z(hidden_states).reshape(batch, seq, -1, attn.head_v_dim)
    b = attn.in_proj_b(hidden_states)
    a = attn.in_proj_a(hidden_states)

    # The causal convolution over the last inputs and the new ones: as causal_conv1d_update, for any number of tokens.
    x = torch.cat([conv_state, mixed], dim=-1)
    new_conv_state = x[:, :, -(attn.conv_kernel_size - 1):]
    mixed = F.silu(F.conv1d(x, attn.conv1d.weight, None, groups=attn.conv_dim)).transpose(1, 2)

    query, key, value = torch.split(mixed, [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
    query = query.reshape(batch, seq, -1, attn.head_k_dim)
    key = key.reshape(batch, seq, -1, attn.head_k_dim)
    value = value.reshape(batch, seq, -1, attn.head_v_dim)
    beta = b.sigmoid()
    g = -attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)
    if attn.num_v_heads // attn.num_k_heads > 1:
        query = query.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)
        key = key.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)
    query, key = l2norm(query), l2norm(key)
    output, new_state = GatedDeltaRule.apply(query, key, value, g, beta, recurrent_state, attn.head_k_dim ** -0.5)

    output = attn.norm(output.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim))
    return attn.out_proj(output.reshape(batch, seq, -1)), new_conv_state, new_state


class KVCache:
    """The update of a transformers Cache, for the full attention layers: the past of the inputs, then the new keys."""

    def __init__(self):
        self.past, self.present = {}, {}

    def update(self, keys, values, layer_idx, *args, **kwargs):
        past_keys, past_values = self.past[layer_idx]
        keys, values = torch.cat([past_keys, keys], dim=-2), torch.cat([past_values, values], dim=-2)
        self.present[layer_idx] = (keys, values)
        return keys, values


def state_names(config):
    """The names of the past inputs, per layer, in their order."""
    config = text_config(config)
    names = []
    for i, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            names += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
        else:
            names += [f"past_conv.{i}", f"past_recurrent.{i}"]
    return names


def empty_states(config, batch=1, past=0, dtype=torch.float32):
    """The states of the first step: empty KV caches and zero linear attention states."""
    config = text_config(config)
    head_dim = config.head_dim
    conv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim + config.linear_num_value_heads * config.linear_value_head_dim
    states = []
    for kind in config.layer_types:
        if kind == "full_attention":
            states += [torch.zeros(batch, config.num_key_value_heads, past, head_dim, dtype=dtype)] * 2
        else:
            states.append(torch.zeros(batch, conv_dim, config.linear_conv_kernel_dim - 1, dtype=dtype))
            states.append(torch.zeros(batch, config.linear_num_value_heads, config.linear_key_head_dim,
                                      config.linear_value_head_dim, dtype=dtype))
    return states


class HybridStep(torch.nn.Module):
    """One step of Qwen3.5: new tokens in, logits and the next states out."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = text_config(model.config)

    def forward(self, input_ids, attention_mask, position_ids, *past):
        text = self.model.model
        hidden = text.embed_tokens(input_ids)
        # The rotary embedding of transformers takes 3 rows of positions for images (mRoPE). For text the rows are
        # the same, and it is the rotary embedding of the first dims of each head.
        rotary = text.rotary_emb
        freqs = position_ids[..., None].float() * rotary.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        position_embeddings = ((emb.cos() * rotary.attention_scaling).to(hidden.dtype),
                               (emb.sin() * rotary.attention_scaling).to(hidden.dtype))
        # The causal mask of transformers uses torch.vmap, which TorchScript cannot trace: pass a 4D additive mask.
        seq, total = input_ids.shape[1], attention_mask.shape[1]
        query = torch.arange(seq) + (total - seq)
        allowed = (torch.arange(total)[None, :] <= query[:, None])[None, None] & attention_mask[:, None, None, :].bool()
        mask = torch.zeros(allowed.shape, dtype=hidden.dtype).masked_fill(~allowed, torch.finfo(hidden.dtype).min)

        cache = KVCache()
        present = []
        for i, layer in enumerate(text.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            if layer.block_type == "linear_attention":
                hidden, conv, recurrent = linear_attention(layer.linear_attn, hidden, past[2 * i], past[2 * i + 1])
                present += [conv, recurrent]
            else:
                cache.past[i] = (past[2 * i], past[2 * i + 1])
                hidden, _ = layer.self_attn(hidden_states=hidden, position_embeddings=position_embeddings,
                                            attention_mask=mask, past_key_values=cache)
                present += list(cache.present[i])
            hidden = residual + hidden
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        logits = self.model.lm_head(text.norm(hidden))
        return (logits, *present)


def export_onnx(model, path):
    config = text_config(model.config)
    names = state_names(config)
    present_names = [n.replace("past_key_values.", "present.").replace("past_", "present_") for n in names]
    past = 2
    states = empty_states(config, past=past)
    # Random states, so that the trace does not fold zeros.
    states = [torch.randn_like(s) for s in states]
    args = (torch.tensor([[11, 12, 13]]), torch.ones(1, past + 3, dtype=torch.int64), torch.arange(past, past + 3).unsqueeze(0), *states)
    axes = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "total"},
            "position_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}}
    for name, out in zip(names, present_names):
        if name.startswith("past_key_values."):
            axes[name] = {0: "batch", 2: "past"}
            axes[out] = {0: "batch", 2: "total"}
        else:
            axes[name] = {0: "batch"}
            axes[out] = {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(HybridStep(model), args, path, input_names=["input_ids", "attention_mask", "position_ids"] + names,
                          output_names=["logits"] + present_names, dynamic_axes=axes, opset_version=17, dynamo=False,
                          custom_opsets={DOMAIN: 1})


class TorchLM:
    """Runs HybridStep in torch, one step at a time, like the Unity package runs the ONNX model."""

    def __init__(self, model):
        self.step = HybridStep(model).eval()
        self.config = text_config(model.config)

    def run(self, input_ids, states=None):
        states = states or empty_states(self.config)
        kinds = self.config.layer_types
        past = next(s.shape[2] for s, kind in zip(states[::2], kinds) if kind == "full_attention")
        seq = len(input_ids)
        with torch.no_grad():
            out = self.step(torch.tensor([input_ids]), torch.ones(1, past + seq, dtype=torch.int64),
                            torch.arange(past, past + seq)[None], *states)
        return out[0], list(out[1:])

    def greedy(self, input_ids, new_tokens):
        logits, states = self.run(input_ids)
        tokens = []
        for _ in range(new_tokens):
            tokens.append(int(logits[0, -1].argmax()))
            logits, states = self.run(tokens[-1:], states)
        return tokens


def check_export(model, new_tokens):
    """Checks that HybridStep computes what the model of transformers computes, for prompt and decode steps."""
    lm = TorchLM(model)
    prompt = list(range(20, 27))
    with torch.no_grad():
        expected = model(torch.tensor([prompt])).logits[0]
    logits, states = lm.run(prompt[:4])
    logits2, states = lm.run(prompt[4:5], states)
    logits3, _ = lm.run(prompt[5:], states)
    actual = torch.cat([logits[0], logits2[0], logits3[0]])
    error = (actual - expected).abs().max().item()
    print(f"max logits error of the export module vs transformers: {error:.2e}")
    assert error < 1e-3 * max(1.0, expected.abs().max().item()), error
    with torch.no_grad():
        generated = model.generate(torch.tensor([prompt]), max_new_tokens=new_tokens, do_sample=False)[0, len(prompt):].tolist()
    assert lm.greedy(prompt, new_tokens) == generated, "greedy generation differs from transformers"
