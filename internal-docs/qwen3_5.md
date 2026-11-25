# Qwen3.5

Qwen3.5 mixes two kinds of layers. Each fourth layer is full attention, with a KV cache like the other models. The other layers are the linear attention of Gated DeltaNet, whose cache is two fixed-size states instead of the keys and values of all tokens.

## The linear attention layers

Each linear attention layer keeps two states that each step replaces:

| State | Shape | What |
|---|---|---|
| conv | [batch, conv channels, conv kernel - 1] | The last inputs of the causal convolution before the delta rule. 6144 x 3 in Qwen3.5-0.8B. |
| recurrent | [batch, value heads, key size, value size] | The state of the gated delta rule. 16 x 128 x 128 in Qwen3.5-0.8B. |

Both start as zeros and stay float32, also in a float16 model, as transformers keeps them (`mamba_ssm_dtype` float32). Their size does not depend on the number of tokens.

`DynamicCache(config)` holds them where `layer_types` says `linear_attention`, as transformers' `DynamicCache(config=...)` holds a `LinearAttentionLayer` there. The indexer of the cache returns the two states of such a layer. `GetSeqLength` reads the first full attention layer, as transformers does.

## The export

`tools/causal_lm/hybrid.py` exports one model for all steps, with the inputs and outputs of the other models, and for each linear attention layer i:

```
past_conv.{i}, past_recurrent.{i}        in
present_conv.{i}, present_recurrent.{i}  out
```

The gated delta rule is a node `GatedDeltaRule` of the ONNX domain `ai.doji`, which tools.doji.onnx-model-loader maps to the layer of tools.doji.sentis-utils. The layer runs any number of tokens in one run, so the prompt and each new token use the same model. ONNX Loop and Scan, the standard way, do not load in Inference Engine.

onnxruntime cannot run the node. `make_models.py` checks the export module in torch against the model of transformers instead, and the tests compare Unity with transformers.

### Differences from transformers

- The rotary embedding of Qwen3.5 has three rows of positions for images (mRoPE). For text the three rows are the same, so the export computes the rotary embedding of the first quarter of each head from the text positions.
- The causal convolution concatenates the conv state and the new inputs, and convolves without padding, as `causal_conv1d_update` does for one token. transformers does this for one token, and pads with zeros for a prompt. Both give the same result when the state starts as zeros.
- The export loads the text model only (`Qwen3_5ForCausalLM`). The vision tower and the multi-token prediction head of the checkpoint are not exported.

## Config

The checkpoint of Qwen3.5 is `Qwen3_5ForConditionalGeneration`, whose `config.json` has the language model in `text_config`. `PretrainedConfig.GetTextConfig` returns it, as transformers' `get_text_config`. `AutoModelForCausalLM` loads both `Qwen3_5ForConditionalGeneration` and `Qwen3_5ForCausalLM` with `Qwen3_5ForCausalLM`, as transformers does. The checkpoint has no `generation_config.json`. `GenerationConfig.FromModelConfig` then takes the end-of-sequence token from `text_config`, as transformers' `GenerationConfig.from_model_config` does.

## Chat template

The package does not run Jinja: `ApplyChatTemplate` recognizes known forms of templates. The template of Qwen3.5 is ChatML, and its generation prompt opens a thinking block. The block is empty (`<think>\n\n</think>\n\n`) unless `enable_thinking` is true, and then it is `<think>\n`. `ApplyChatTemplate(..., enableThinking)` adds this block when the template has the test `enable_thinking is defined and enable_thinking is true`. The template also trims the content of each message and reformats earlier assistant messages that have a thinking block. `ApplyChatTemplate` does not do these.

## Tokenizer

transformers v5 loads the `tokenizer.json` of the model type `qwen3_5` as it is (`Qwen3_5Tokenizer`), and `AutoTokenizer` does the same with `TokenizersBackend`. For the model type `qwen3_5_text` of the tiny test model, both load the hub class `Qwen2Tokenizer`. The two give the same ids.
