# AGENTS.md

`tools.doji.transformers`: transformer models for Unity Inference Engine, ported from Hugging Face transformers: tokenizers, chat templates, and text generation with decoder-only language models (SmolLM2, Phi-3).

Unity 6000.6.4f1 for development; `tools.doji.transformers/package.json` sets the minimum supported version.

## Layout

From the companion project, the repository root is `../..`.

- `tools.doji.transformers/`: the UPM package.
  - `Runtime/`: assembly `Doji.Transformers`, namespace `Doji.AI.Transformers`. Players compile it, so editor code goes in `Editor/`.
  - `Tests/Editor/`: EditMode tests. They compare tiny models with onnxruntime and transformers; the reference data comes from `tools/causal_lm/make_models.py`.
  - `Samples~/`: the text generation sample.
- `projects/Transformers/`: the companion project. It references the package with a local `file:` path. Open, test, and build from here. `Assets/SampleCheck/` holds git-ignored measurement probes.
- `tools/`: Python scripts that export models and make reference data.
- `internal-docs/`: maintainer notes. Start at `internal-docs/README.md`. The Inference Engine features this package builds on are described in `internal-docs/` of `G:\Unity\tools.doji.sentis-utils`.

## Package direction

- Follow Hugging Face transformers: class names, defaults, and behavior match it unless Unity forces a difference. Write down such a difference in `internal-docs/`.
- Memory and speed features build on `tools.doji.sentis-utils`. Make Inference Engine-level changes there, not here.

## Measuring

- Follow the `model-perf-run` skill for any model load, benchmark, or RAM/VRAM measurement: the RAM watchdog, one Unity instance at a time, headless runs, nvidia-smi for VRAM.
- Run only one Unity instance at a time on this machine. Never run heavy work in the user's open Editor.
- Per-machine numbers go in the README performance table, with date, Unity version, graphics API and hardware.

## Changelog

Follow the `doji-changelog` skill. No figures measured on one machine in the changelog or in commit messages.

## Validation

- Preserve `.meta` files when you move or add Unity assets and scripts.
- Run the EditMode tests in the companion project after a change to generation, the cache, or a tokenizer. Test models come from AI Model Hub; a test that fails because a model is missing from the cache is not a regression.
