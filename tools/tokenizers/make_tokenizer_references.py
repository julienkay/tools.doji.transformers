"""
Writes the reference data of the tokenizer tests of the Unity transformers package: what the official
transformers tokenizers return for many edge cases.

The tests load the same tokenizer files through AI Model Hub from the tiny test model repository
(TEST_REPO), which has the SmolLM2 tokenizer at its root and the other tokenizers in tokenizers/<name>/.
AI Model Hub only takes repositories with a model file, so the tokenizer files ride along with the tiny model.
`copy` puts the files of the revisions below into a folder to upload there with tools/causal_lm/make_models.py upload.

Usage:
  python make_tokenizer_references.py copy <tiny model folder>
  python make_tokenizer_references.py

Needs transformers 5.x (pip install "transformers>=5" sentencepiece protobuf).
"""
import sys
import json
import os

import transformers
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.normpath(os.path.join(HERE, "..", "..", "tools.doji.transformers", "Tests", "Editor", "Resources", "tokenizers"))

# name: repo, revision. Made with transformers 5.x, whose tokenizers all run on the tokenizers library.
TOKENIZERS = {
    "smollm2": ("julienkay/tiny-random-LlamaForCausalLM-onnx", "1675db936b735d669df5a25bb238179b110486e2"),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "qwen2_5": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "phi3": ("microsoft/Phi-3-mini-4k-instruct", "f39ac1d28e925b323eae81227eaba4464caced4e"),
    "clip": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
}

# Calls with pairs, batches, truncation and padding: the keyword arguments of tokenizer(...).
CALLS = [
    {"text": "Hello world!", "text_pair": "How are you?"},
    {"text": "Hello world!", "max_length": 6, "truncation": True},
    {"text": "Hello world!", "text_pair": "How are you doing today, my friend?", "max_length": 9, "truncation": True},
    {"text": "Hello world! How are you?", "text_pair": "Fine.", "max_length": 7, "truncation": "only_first"},
    {"text": "a photo of a cat", "padding": "max_length", "max_length": 12},
    {"text": "a photo of a cat", "padding": "max_length", "max_length": 4, "truncation": True},
    {"text": ["a photo of a cat", "a dog", ""], "padding": True},
    {"text": ["a photo of a cat", "a dog"], "padding": True, "pad_to_multiple_of": 8},
    {"text": ["first text", "second"], "text_pair": ["pair one", "pair two is longer"], "padding": "longest"},
    {"text": "Hello world!", "add_special_tokens": False, "return_special_tokens_mask": True},
    {"text": "Hello world!", "return_special_tokens_mask": True, "return_token_type_ids": True},
]

TEXTS = [
    # whitespace
    "", " ", "   ", "\n", "\n\n", "\t", "\r\n", " \n ", "a", " a", "a ", "  a  b   c  ", "line1\nline2\n\nline3",
    "   \n\n  indented", "trailing spaces   ", "tab\tseparated\tvalues",
    # words and case
    "Hello world!", "Hello  world", "HeLLo WoRLD", "ALLCAPS TEXT", "The quick brown fox jumps over the lazy dog.",
    "supercalifragilisticexpialidocious" * 3, "a" * 64, "!" * 20, "ab" * 30,
    # contractions
    "I'm sure you're right, we've seen it, they'll come, he'd go, it's fine.", "I'M DON'T YOU'RE", "can't won't shan't y'all",
    "'quoted' and \"double quoted\"", "rock 'n' roll",
    # numbers
    "1234567890", "3.14159", "1,000,000", "v2.0.1", "2024-11-23", "x² + y³", "Ⅻ chapters", "١٢٣ ٤٥٦", "0x1F 0b1010 1e-5",
    "The year 2024 had 365 days.", "Call 555-0100 at 10:30am.",
    # punctuation and symbols
    "Hello, world... !!! ??? --- ***", "(a) [b] {c} <d>", "user@example.com", "https://huggingface.co/docs?x=1&y=2#top",
    "#hashtag @mention $100 50% a&b", "\\path\\to\\file C:/Users/name", "~`^|_=+",
    # code
    "def f(x):\n    return x**2\n", "if (a && b) { c++; } else { d--; }", "<html><body>Hi</body></html>",
    "SELECT * FROM t WHERE id = 1;", "    four spaces\n\ttab\n        eight",
    # unicode
    "Größe, naïve café, Ångström", "東京は日本の首都です。", "한국어 텍스트입니다", "Привет, мир!", "مرحبا بالعالم",
    "हिन्दी पाठ", "Ελληνικά", "ไทย", "😀😃🎉👍🏽", "👨‍👩‍👧‍👦 family", "e\u0301 vs \u00e9", "ﬁ ligature", "a\u200bb zero width",
    "a\u00a0b non-breaking", "en\u2013dash em\u2014dash", "“curly” ‘quotes’", "½ ¼ ¾ €", "\u2028line separator",
    "mixed 中文 and English 123",
]

# Special tokens of each tokenizer, to add cases around them.
SPECIAL_TEXTS = {
    "smollm2": ["<|im_start|>user\nhi<|im_end|>\n", "<|endoftext|>", "<|im_end|><|im_end|>", "<|im_start", "abc<|im_end|>def",
                " <|im_end|> ", "<|im_start|>assistant\n<|im_end|>"],
    "gpt2": ["<|endoftext|>", "a<|endoftext|>b", " <|endoftext|> ", "<|endoftext|><|endoftext|>", "<|endoftext"],
    "qwen2_5": ["<|im_start|>user\nhi<|im_end|>\n", "<|endoftext|>", "<|im_end|><|im_end|>", "abc<|im_end|>def", " <|im_end|> "],
    "phi3": ["<|user|>\nhi<|end|>\n<|assistant|>\n", "<|endoftext|>", "a<|end|>b", " <|end|> ", "<s>hi</s>"],
    "clip": ["<|startoftext|>a photo<|endoftext|>"],
}

CHATS = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "Hi!"}],
    [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello! How can I help?"},
     {"role": "user", "content": "Tell me a joke.\nA short one."}],
]


def call(tokenizer, kwargs):
    try:
        encoding = tokenizer(**kwargs)
    except Exception as e:  # for example padding without a pad token
        return {"kwargs": kwargs, "error": type(e).__name__}
    return {"kwargs": kwargs, "output": {k: v for k, v in encoding.items()}}


# name: repo, revision of tokenizers loaded by LlamaTokenizer from tokenizer.model alone, without tokenizer.json.
SENTENCEPIECE = {
    "phi3_spm": ("microsoft/Phi-3-mini-4k-instruct", "f39ac1d28e925b323eae81227eaba4464caced4e"),
}


def load(name, repo, revision):
    if name not in SENTENCEPIECE:
        return AutoTokenizer.from_pretrained(repo, revision=revision)
    import tempfile
    import shutil
    from huggingface_hub import hf_hub_download
    from transformers import LlamaTokenizer
    folder = tempfile.mkdtemp()
    for file in ["tokenizer.model", "tokenizer_config.json"]:
        shutil.copy(hf_hub_download(repo, file, revision=revision), os.path.join(folder, file))
    return LlamaTokenizer.from_pretrained(folder)


def references(name, repo, revision):
    from transformers import AutoConfig
    tokenizer = load(name, repo, revision)
    model_type = AutoConfig.from_pretrained(repo, revision=revision).model_type
    cases = []
    for text in TEXTS + SPECIAL_TEXTS.get(name.removesuffix("_spm"), []):
        ids = tokenizer.encode(text)
        cases.append({
            "text": text,
            "ids": ids,
            "decoded": tokenizer.decode(ids),
            "decoded_skip_special": tokenizer.decode(ids, skip_special_tokens=True),
            # Each token alone, which cuts characters that span several tokens.
            "decoded_each": [tokenizer.decode([i]) for i in ids],
        })
    chats = []
    if tokenizer.chat_template:
        for messages in CHATS:
            for prompt in (True, False):
                chats.append({"messages": messages, "add_generation_prompt": prompt,
                              "text": tokenizer.apply_chat_template(messages, add_generation_prompt=prompt, tokenize=False)})
    subfolder = f"tokenizers/{name.removesuffix('_spm')}/" if name.removesuffix("_spm") in FILES else ""
    return {"source": repo, "revision": revision, "repo": TEST_REPO, "subfolder": subfolder,
            "transformers": transformers.__version__, "model_type": model_type, "class": type(tokenizer).__name__,
            "cases": cases, "calls": [call(tokenizer, kwargs) for kwargs in CALLS], "chats": chats}


TEST_REPO = "julienkay/tiny-random-LlamaForCausalLM-onnx"
FILES = {
    "gpt2": ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    "qwen2_5": ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    "phi3": ["tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"],
    "clip": ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
}


def copy(out):
    import shutil
    from huggingface_hub import hf_hub_download
    for name, files in FILES.items():
        repo, revision = TOKENIZERS[name]
        folder = os.path.join(out, "tokenizers", name)
        os.makedirs(folder, exist_ok=True)
        for file in files:
            shutil.copy(hf_hub_download(repo, file, revision=revision), os.path.join(folder, file))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "copy":
        copy(sys.argv[2])
        return
    os.makedirs(RESOURCES, exist_ok=True)
    for name, (repo, revision) in {**TOKENIZERS, **SENTENCEPIECE}.items():
        data = references(name, repo, revision)
        with open(os.path.join(RESOURCES, name + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(name, data["class"], len(data["cases"]), "cases", len(data["chats"]), "chats")


if __name__ == "__main__":
    main()
