"""
Exports Llama-style causal language models to plain ONNX for Unity Inference Engine, builds the tiny test
model of the Unity transformers package, and writes its reference data.

The official ONNX exports on the Hugging Face Hub (for example onnx/model.onnx of SmolLM2) use com.microsoft
contrib ops (GroupQueryAttention, RotaryEmbedding, SkipSimplifiedLayerNormalization) that Unity Inference
Engine cannot import. This script exports with torch.onnx (TorchScript, opset 17), which writes standard ops only.

One model runs every step, the prompt and each new token, with the inputs and outputs of optimum exports:

  inputs   input_ids [batch, seq] int64, attention_mask [batch, past + seq] int64, position_ids [batch, seq] int64,
           past_key_values.{i}.key / past_key_values.{i}.value [batch, kv_heads, past, head_dim] float32
  outputs  logits [batch, seq, vocab], present.{i}.key / present.{i}.value [batch, kv_heads, past + seq, head_dim]

The tiny test models have random weights in the architecture of a real model (Llama, Phi-3) and its tokenizer,
so the tests run the tokenizer files of a real model.

Usage:
  python make_models.py export <model id or folder> <out folder> [--dtype float16]
  python make_models.py tiny <out folder> [--arch llama|phi3]
  python make_models.py references <tiny onnx folder> [--arch llama|phi3]
  python make_models.py upload <folder> <repo id>
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.normpath(os.path.join(HERE, "..", "..", "tools.doji.transformers", "Tests", "Editor", "Resources"))

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
# Other tokenizer files that some models have, for example the sentencepiece model of Phi-3.
EXTRA_TOKENIZER_FILES = ["tokenizer.model", "added_tokens.json"]

# The tiny test models: random weights in the architecture of a real model, with its tokenizer. "prefix" names
# the reference files of the tests.
TINY = {
    "llama": dict(prefix="tiny_llama", tokenizer=("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac")),
    "phi3": dict(prefix="tiny_phi3", tokenizer=("microsoft/Phi-3-mini-4k-instruct", "f39ac1d28e925b323eae81227eaba4464caced4e")),
}


def tiny_config(arch):
    # Large weights spread the logits, so greedy decoding has clear winners and does not flip on rounding.
    if arch == "llama":
        from transformers import LlamaConfig
        return LlamaConfig(
            vocab_size=49152, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=8192, rope_theta=100000, rms_norm_eps=1e-5,
            tie_word_embeddings=True, bos_token_id=1, eos_token_id=2, pad_token_id=2, initializer_range=0.5)
    from transformers import Phi3Config
    return Phi3Config(
        vocab_size=32064, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=4096, original_max_position_embeddings=4096, rope_theta=10000,
        rms_norm_eps=1e-5, tie_word_embeddings=False, bos_token_id=1, eos_token_id=32000, pad_token_id=32000,
        initializer_range=0.5)

# Texts for the tokenizer tests: digits, contractions, whitespace runs, unicode, emoji and special tokens.
TEXTS = [
    "Hello world!",
    "The year 2024 had 365 days, didn't it?",
    "  leading spaces,\ttabs\nand new lines\n\n",
    "Größe, naïve café, 東京, 😀",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n",
]
MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]
NEW_TOKENS = 8


def load_model(model_id, dtype=torch.float32):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="eager").eval()


class Step(torch.nn.Module):
    """One step of a causal LM: new tokens in, logits and the grown KV cache out."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids, *past):
        from transformers import DynamicCache
        cache = DynamicCache()
        for i in range(self.layers):
            cache.update(past[2 * i], past[2 * i + 1], i)
        # The causal mask of transformers uses torch.vmap, which TorchScript cannot trace: pass a 4D additive mask.
        seq, total = input_ids.shape[1], attention_mask.shape[1]
        query = torch.arange(seq) + (total - seq)
        allowed = (torch.arange(total)[None, :] <= query[:, None])[None, None] & attention_mask[:, None, None, :].bool()
        dtype = self.model.dtype
        mask = torch.zeros(allowed.shape, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)
        out = self.model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                         past_key_values=cache, use_cache=True)
        present = []
        for i in range(self.layers):
            layer = out.past_key_values.layers[i]
            present += [layer.keys, layer.values]
        return (out.logits, *present)


def kv_shape(config):
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return config.num_key_value_heads, head_dim


def export_onnx(model, path):
    config = model.config
    layers = config.num_hidden_layers
    heads, head_dim = kv_shape(config)
    past, seq = 2, 3
    past_kv = [torch.randn(1, heads, past, head_dim, dtype=model.dtype) for _ in range(2 * layers)]
    args = (torch.tensor([[11, 12, 13]]), torch.ones(1, past + seq, dtype=torch.int64),
            torch.arange(past, past + seq).unsqueeze(0), *past_kv)
    past_names = [f"past_key_values.{i}.{kind}" for i in range(layers) for kind in ("key", "value")]
    present_names = [f"present.{i}.{kind}" for i in range(layers) for kind in ("key", "value")]
    axes = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "total"},
            "position_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}}
    axes.update({name: {0: "batch", 2: "past"} for name in past_names})
    axes.update({name: {0: "batch", 2: "total"} for name in present_names})
    folder = os.path.dirname(os.path.abspath(path))
    before = set(os.listdir(folder))
    with torch.no_grad():
        torch.onnx.export(Step(model), args, path, input_names=["input_ids", "attention_mask", "position_ids"] + past_names,
                          output_names=["logits"] + present_names, dynamic_axes=axes, opset_version=17, dynamo=False)
    # Above 2 GB, torch writes each weight to a file of its own. Pack them into one model.onnx_data, which AI Model
    # Hub takes along with model.onnx.
    weights = set(os.listdir(folder)) - before - {os.path.basename(path)}
    if weights:
        import onnx
        onnx.save_model(onnx.load(path), path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=os.path.basename(path) + "_data", size_threshold=1024)
        for name in weights:
            os.remove(os.path.join(folder, name))


class OnnxLM:
    """Runs an exported model in onnxruntime, one step at a time, like the Unity package."""

    def __init__(self, path, config):
        import onnxruntime
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.layers = config.num_hidden_layers
        self.heads, self.head_dim = kv_shape(config)

    def run(self, input_ids, past=None):
        past = past or [np.zeros((1, self.heads, 0, self.head_dim), np.float32)] * (2 * self.layers)
        p, s = past[0].shape[2], len(input_ids)
        feed = {"input_ids": np.array([input_ids], np.int64), "attention_mask": np.ones((1, p + s), np.int64),
                "position_ids": np.arange(p, p + s, dtype=np.int64)[None]}
        feed.update({f"past_key_values.{i}.{kind}": past[2 * i + j] for i in range(self.layers)
                     for j, kind in enumerate(("key", "value"))})
        out = self.session.run(None, feed)
        return out[0], out[1:]

    def greedy(self, input_ids, new_tokens):
        logits, past = self.run(input_ids)
        tokens = []
        for _ in range(new_tokens):
            tokens.append(int(logits[0, -1].argmax()))
            logits, past = self.run(tokens[-1:], past)
        return tokens


def check_export(model, path):
    """Checks that the ONNX model computes what the torch model computes, for prompt and decode steps."""
    lm = OnnxLM(path, model.config)
    prompt = list(range(20, 27))
    with torch.no_grad():
        expected = model(torch.tensor([prompt])).logits[0].numpy()
    logits, past = lm.run(prompt[:4])
    logits2, _ = lm.run(prompt[4:], past)
    actual = np.concatenate([logits[0], logits2[0]])
    error = np.abs(actual - expected).max()
    print(f"max logits error vs torch: {error:.2e}")
    assert error < 1e-3 * max(1.0, np.abs(expected).max()), error
    with torch.no_grad():
        generated = model.generate(torch.tensor([prompt]), max_new_tokens=NEW_TOKENS, do_sample=False)[0, len(prompt):].tolist()
    assert lm.greedy(prompt, NEW_TOKENS) == generated, "greedy generation differs from torch"


def copy_files(model_id, out, files):
    from huggingface_hub import hf_hub_download
    for name in files:
        if os.path.isdir(model_id):
            src = os.path.join(model_id, name)
            if not os.path.exists(src):
                continue
        else:
            try:
                src = hf_hub_download(model_id, name)
            except Exception:
                continue
        shutil.copy(src, os.path.join(out, name))


def export(args):
    os.makedirs(args.out, exist_ok=True)
    # float16 halves the memory of the export, for models of a few billion parameters. The weights, the KV cache
    # and the logits are then float16 too. onnxruntime on the CPU does not run every float16 op: check those in Unity.
    float16 = getattr(args, "dtype", "float32") == "float16"
    model = load_model(args.model, torch.float16 if float16 else torch.float32)
    path = os.path.join(args.out, "model.onnx")
    export_onnx(model, path)
    copy_files(args.model, args.out, ["config.json", "generation_config.json"] + TOKENIZER_FILES + EXTRA_TOKENIZER_FILES)
    if not float16:
        check_export(model, path)
    print("exported", path)


def tiny(args):
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, GenerationConfig
    torch.manual_seed(0)
    config = tiny_config(args.arch)
    model = AutoModelForCausalLM.from_config(config).eval()
    torch_dir = os.path.join(args.out, "torch")
    model.save_pretrained(torch_dir)
    GenerationConfig(bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
                     pad_token_id=config.pad_token_id).save_pretrained(torch_dir)
    repo, revision = TINY[args.arch]["tokenizer"]
    for name in TOKENIZER_FILES + EXTRA_TOKENIZER_FILES:
        try:
            src = hf_hub_download(repo, name, revision=revision)
        except Exception:
            continue
        shutil.copy(src, os.path.join(torch_dir, name))
    export(argparse.Namespace(model=torch_dir, out=os.path.join(args.out, "onnx")))
    margins(load_model(torch_dir))


def margins(model):
    """Prints the smallest gap between the best and second-best logit while generating, to see that greedy is stable."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    ids = tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=NEW_TOKENS, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    gaps = [float(s[0].topk(2).values.diff().abs()) for s in out.scores]
    print("smallest top-2 logit gap:", min(gaps))


def write(name, values):
    os.makedirs(RESOURCES, exist_ok=True)
    with open(os.path.join(RESOURCES, name + ".txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(repr(float(v)) if isinstance(v, float) else str(v) for v in values))


def references(args):
    """Reference data of the tests: the tokenizer of transformers, and onnxruntime on the exact ONNX file Unity loads."""
    from transformers import AutoTokenizer
    root = args.tiny
    prefix = TINY[args.arch]["prefix"]
    tokenizer = AutoTokenizer.from_pretrained(root)
    cases = []
    for text in TEXTS:
        ids = tokenizer.encode(text)
        cases.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids), "decoded_skip_special": tokenizer.decode(ids, skip_special_tokens=True)})
    chat = tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=False)
    with open(os.path.join(RESOURCES, prefix + "_tokenizer.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"cases": cases, "messages": MESSAGES, "chat": chat}, f, ensure_ascii=False, indent=1)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(root)
    lm = OnnxLM(os.path.join(root, "model.onnx"), config)
    prompt = tokenizer.encode(chat)
    logits, _ = lm.run(prompt)
    write(prefix + "_prompt_ids", prompt)
    write(prefix + "_last_logits", logits[0, -1, :64].tolist())
    generated = lm.greedy(prompt, NEW_TOKENS)
    write(prefix + "_greedy", generated)
    if os.path.exists(os.path.join(root, "..", "torch")):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(os.path.join(root, "..", "torch"))
        with torch.no_grad():
            expected = model.generate(torch.tensor([prompt]), max_new_tokens=NEW_TOKENS, do_sample=False)[0, len(prompt):].tolist()
        assert generated == expected, (generated, expected)
    print("wrote references to", RESOURCES, "greedy:", generated)


def upload(args):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, exist_ok=True)
    info = api.upload_folder(repo_id=args.repo, folder_path=args.folder, commit_message="Upload plain ONNX export for Unity")
    print("commit", info.oid)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export"); p.add_argument("model"); p.add_argument("out")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32"); p.set_defaults(run=export)
    p = sub.add_parser("tiny"); p.add_argument("out"); p.add_argument("--arch", choices=list(TINY), default="llama")
    p.set_defaults(run=tiny)
    p = sub.add_parser("references"); p.add_argument("tiny"); p.add_argument("--arch", choices=list(TINY), default="llama")
    p.set_defaults(run=references)
    p = sub.add_parser("upload"); p.add_argument("folder"); p.add_argument("repo"); p.set_defaults(run=upload)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
