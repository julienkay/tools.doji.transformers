<a href="https://www.doji-tech.com/">
  <img src="https://www.doji-tech.com/assets/favicon.ico" alt="doji logo" title="Doji" align="right" height="70" />
</a>

# Transformers
A Unity package to run pretrained transformer models with Unity Inference Engine

[OpenUPM] · [Documentation (coming soon)] · [Feedback/Questions]

## About

This is essentially a C# port of Hugging Face’s [transformers] library.

There are two use cases for this package right now:
- It's used by the [tools.doji.diffusers] package to run Stable Diffusion models in Unity (most SD models use a ClipTokenizer for prompting, newer pipelines require additional ones like T5Tokenizer)
- To run small LLMs like [SmolLM2](https://huggingface.co/julienkay/SmolLM2-135M-Instruct-onnx) in Unity

### Installation

<details>
  <summary> via OpenUPM </summary>
  
 1. In `Edit -> Project Settings -> Package Manager`, add a new scoped registry:

        Name: Doji
        URL: https://package.openupm.com
        Scope(s): com.doji
 
  4. In the Package Manager install `tools.doji.transformers` either by name or select it in the list under `Package Manager -> My Registries`

</details>

### Usage

Models come through [AI Model Hub] (`tools.doji.model-hub`). List them in the project's AI Model Hub manifest (`Project Settings > Doji > AI > AI Model Hub`).

```csharp
using Doji.AI.Transformers;

var tokenizer = await AutoTokenizer.FromPretrainedAsync("julienkay/SmolLM2-135M-Instruct-onnx");
var model = await AutoModelForCausalLM.FromPretrainedAsync("julienkay/SmolLM2-135M-Instruct-onnx");

string prompt = tokenizer.ApplyChatTemplate(new[] { ChatMessage.User("What is the capital of France?") });
int[] inputIds = tokenizer.Encode(prompt).InputIds.ToArray();
var streamer = new TextStreamer(tokenizer, text => Debug.Log(text), skipPrompt: true);
int[] outputIds = await model.GenerateAsync(inputIds, new GenerationConfig { MaxNewTokens = 64 }, streamer);
```

`FromPretrained` and `Generate` (without `Async`) run without waiting. They work in the Editor and for bundled or already downloaded models on Windows, macOS, Linux and iOS.

The models are plain ONNX exports with the input and output names of optimum (`input_ids`, `attention_mask`, `position_ids`, `past_key_values.*`). `tools/causal_lm/make_models.py` exports a transformers model this way.

## Roadmap:

Tokenizers
- [x] CLIPTokenizer
- [x] LLamaTokenizer
- [x] GPT2Tokenizer
- [x] PreTrainedTokenizerFast (byte-level BPE `tokenizer.json`)
- [ ] T5Tokenizer

LLMs
- [x] Llama (SmolLM2)
- [ ] Phi-3

### Dependencies
- [AI Model Hub] (`tools.doji.model-hub`)
- [tools.doji.sentencepiece]

[OpenUPM]: https://openupm.com/packages/tools.doji.transformers
[Documentation (coming soon)]: https://github.com/julienkay/tools.doji.transformers
[Feedback/Questions]: https://discussions.unity.com/t/stable-diffusion-diffusers-transformers-package/332701?u=julienkay
[transformers]: https://github.com/huggingface/transformers
[tools.doji.diffusers]: https://github.com/julienkay/tools.doji.diffusers
[AI Model Hub]: https://docs.doji-tech.com/tools.doji.model-hub
[tools.doji.sentencepiece]: https://github.com/julienkay/tools.doji.sentencepiece
