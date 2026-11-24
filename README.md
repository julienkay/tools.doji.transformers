<a href="https://www.doji-tech.com/">
  <img src="https://www.doji-tech.com/assets/favicon.ico" alt="doji logo" title="Doji" align="right" height="70" />
</a>

# Transformers
A Unity package to run pretrained transformer models with Unity Inference Engine

[OpenUPM] · [Documentation (coming soon)] · [Feedback/Questions]

## About

This is essentially a C# port of Hugging Face’s [transformers] library.

There are two use cases for this package right now:
- It's used by the [tools.doji.diffusers] package to run Stable Diffusion models in Unity (most SD models use a CLIPTokenizer for prompting, newer pipelines require additional ones like T5Tokenizer)
- To run small LLMs like [SmolLM2](https://huggingface.co/julienkay/SmolLM2-135M-Instruct-sentis-float16) in Unity

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

var tokenizer = await AutoTokenizer.FromPretrainedAsync("julienkay/SmolLM2-135M-Instruct-sentis-float16");
var model = await AutoModelForCausalLM.FromPretrainedAsync("julienkay/SmolLM2-135M-Instruct-sentis-float16");

string prompt = tokenizer.ApplyChatTemplate(new[] { ChatMessage.User("What is the capital of France?") });
int[] inputIds = tokenizer.Encode(prompt, addSpecialTokens: false).InputIds.ToArray();
var streamer = new TextStreamer(tokenizer, text => Debug.Log(text), skipPrompt: true);
int[] outputIds = await model.GenerateAsync(inputIds, new GenerationConfig { MaxNewTokens = 64 }, streamer);
```

To try it, import the **Text Generation Sample** from the Package Manager and open its scene. The sample lists its model in a Model Hub manifest asset, so Model Hub downloads the model after the import.

`FromPretrained` and `Generate` (without `Async`) run without waiting. They work in the Editor and for bundled or already downloaded models on Windows, macOS, Linux and iOS.

The models are plain ONNX exports with the input and output names of optimum (`input_ids`, `attention_mask`, `position_ids`, `past_key_values.*`). `tools/causal_lm/make_models.py` exports a transformers model this way.

### Performance

Rough numbers for greedy generation in Play mode with one frame after each token, as the sample runs. Each model generates 64 tokens after a short chat prompt.

| Model | Tokens/s<br>D3D12 / D3D11 | Peak VRAM<br>D3D12 / D3D11 | Peak RAM<br>D3D12 / D3D11 |
|---|---|---|---|
| [SmolLM2-135M-Instruct-sentis-float16](https://huggingface.co/julienkay/SmolLM2-135M-Instruct-sentis-float16) | 13.2 / 21.8 | 0.5 / 0.5 GB | +0.3 / +0.3 GB |
| [SmolLM2-360M-Instruct-sentis-float16](https://huggingface.co/julienkay/SmolLM2-360M-Instruct-sentis-float16) | 11.2 / 19.0 | 1.1 / 1.0 GB | +0.4 / +0.3 GB |
| [SmolLM2-1.7B-Instruct-sentis-float16](https://huggingface.co/julienkay/SmolLM2-1.7B-Instruct-sentis-float16) | 10.2 / 11.0 | 4.0 / 4.0 GB | +0.5 / +0.6 GB |
| [Phi-3-mini-4k-instruct-sentis-uint8](https://huggingface.co/julienkay/Phi-3-mini-4k-instruct-sentis-uint8) | 6.7 / 7.4 | 4.4 / 4.5 GB | +0.3 / +0.4 GB |
| [Phi-3-mini-4k-instruct-sentis-float16](https://huggingface.co/julienkay/Phi-3-mini-4k-instruct-sentis-float16) | 6.0 / 6.5 | 7.8 / 7.7 GB | +0.5 / +0.6 GB |

Long prompts with Phi-3-mini-4k-instruct-sentis-float16, 4 new tokens after the prompt:

| Prompt | D3D12 | D3D11 |
|---|---|---|
| 1024 tokens | 3.6 tokens/s, 10.3 GB VRAM | 4.2 tokens/s, 9.4 GB VRAM |
| 2048 tokens | 2.1 tokens/s, 11.6 GB VRAM (at the limit of the GPU) | 3.0 tokens/s, 9.7 GB VRAM |
| 3072 tokens | not run: 2048 tokens already fill the GPU | 2.2 tokens/s, 10.3 GB VRAM (11.5 GB peak) |

- Peak VRAM is the GPU memory that the model adds, from `nvidia-smi`. In the long-prompt table, VRAM is the memory of the whole GPU while the model holds its cache.
- Peak RAM is the resident memory (working set) that the Editor process adds while it loads and runs the model.
- D3D12 is the default graphics API of Unity 6 on Windows. On D3D12, GPU memory that a model frees during a long prompt can't be reused in the same frame, so long prompts need more VRAM. To use D3D11, turn off *Auto Graphics API for Windows* in the Player settings and put Direct3D11 first, or start the Editor with `-force-d3d11`.
- Measured on 2026-11-24 with `BackendType.GPUCompute`, Unity 6000.6.4f1, an NVIDIA GeForce RTX 3060 (12 GB) and an Intel Core i7-4790K. The D3D11 numbers come from one run per model.

## Roadmap:

Tokenizers
- [x] TokenizersBackend (any `tokenizer.json`, as in transformers v5)
- [x] CLIPTokenizer
- [x] LlamaTokenizer
- [x] GPT2Tokenizer
- [x] Qwen2Tokenizer

LLMs
- [x] Llama (SmolLM2)
- [x] Phi-3

### Dependencies
- [AI Model Hub] (`tools.doji.model-hub`)
- [Tokenizers] (`tools.doji.tokenizers`)

[OpenUPM]: https://openupm.com/packages/tools.doji.transformers
[Documentation (coming soon)]: https://github.com/julienkay/tools.doji.transformers
[Feedback/Questions]: https://discussions.unity.com/t/stable-diffusion-diffusers-transformers-package/332701?u=julienkay
[transformers]: https://github.com/huggingface/transformers
[tools.doji.diffusers]: https://github.com/julienkay/tools.doji.diffusers
[AI Model Hub]: https://docs.doji-tech.com/tools.doji.model-hub
[Tokenizers]: https://github.com/julienkay/tools.doji.tokenizers
