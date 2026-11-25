using Doji.Downloads;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Loads the tokenizer of a pretrained model, choosing the class as transformers v5 does: from the model type of
    /// <c>config.json</c> and the <c>tokenizer_class</c> of <c>tokenizer_config.json</c>.
    /// </summary>
    public static class AutoTokenizer {

        /// <summary>The tokenizer classes by model type, of the models this package has a class for.</summary>
        private static readonly Dictionary<string, string> TokenizerMapping = new() {
            ["clip"] = "CLIPTokenizer",
            ["clip_text_model"] = "CLIPTokenizer",
            ["gpt2"] = "GPT2Tokenizer",
            ["llama"] = "LlamaTokenizer",
            ["qwen2"] = "Qwen2Tokenizer",
            ["qwen2_moe"] = "Qwen2Tokenizer",
            ["qwen3"] = "Qwen2Tokenizer",
            ["qwen3_moe"] = "Qwen2Tokenizer",
        };

        /// <summary>
        /// The model types whose hub tokenizer class is wrong: transformers loads their <c>tokenizer.json</c> as it is,
        /// unless it has a class for the model type.
        /// </summary>
        private static readonly HashSet<string> ModelsWithIncorrectHubTokenizerClass = new() {
            "arctic", "chatlm", "ernie4_5", "ernie4_5_moe", "fuyu", "internlm2", "internlm3", "jamba", "llava",
            "llava_next", "minicpm3", "minicpmv", "nemotron", "phi3", "phi3_v", "phimoe", "qwen2", "qwen3_5", "qwen3_5_moe",
            "qwen3_5_moe_text", "smolvlm", "vipllava",
        };

        /// <summary>
        /// Loads the tokenizer of a model through AI Model Hub. Works on every platform.
        /// </summary>
        public static async Task<PreTrainedTokenizerBase> FromPretrainedAsync(string modelId,
            IProgress<DownloadProgress> progress = null, CancellationToken cancellationToken = default) {
            return await FromPretrainedAsync(await ModelRepository.OpenAsync(modelId, progress, cancellationToken));
        }

        /// <summary>
        /// Loads the tokenizer of a model through AI Model Hub without waiting. Works in the Editor and for bundled
        /// or already downloaded models on Windows, macOS, Linux, and iOS.
        /// </summary>
        public static PreTrainedTokenizerBase FromPretrained(string modelId) {
            return SyncTask.Run(FromPretrainedAsync(ModelRepository.Open(modelId)));
        }

        /// <summary>
        /// Loads the tokenizer of a GGUF file of llama.cpp from its metadata, as transformers'
        /// <c>from_pretrained(..., gguf_file=...)</c> does: the vocabulary, the merges, the special tokens and the
        /// chat template. Supported: byte-level BPE tokenizers (<c>tokenizer.ggml.model</c> gpt2) with the
        /// pre-tokenizers <c>qwen35</c>, <c>qwen2</c> and <c>smollm</c>, and sentencepiece BPE tokenizers
        /// (<c>tokenizer.ggml.model</c> llama, for example of Phi-3) as a <see cref="LlamaTokenizer"/>.
        /// </summary>
        public static PreTrainedTokenizerBase FromGguf(string path) {
            return GgufTokenizer.Load(Doji.Gguf.GgufFile.Open(path));
        }

        /// <param name="subfolder">The folder of the tokenizer files in the repository, such as <c>"tokenizer/"</c>.</param>
        internal static Task<PreTrainedTokenizerBase> FromPretrainedAsync(ModelRepository files, string subfolder = "") {
            return FromPretrainedAsync(files, subfolder, null);
        }

        /// <param name="modelType">The model type, when the folder has no config.json with it.</param>
        internal static async Task<PreTrainedTokenizerBase> FromPretrainedAsync(ModelRepository files, string subfolder, string modelType) {
            string Path(string name) => subfolder + name;
            var config = files.Has(Path(PreTrainedTokenizerBase.TOKENIZER_CONFIG_FILE))
                ? TokenizerConfig.Deserialize(await files.ReadTextAsync(Path(PreTrainedTokenizerBase.TOKENIZER_CONFIG_FILE)))
                : new TokenizerConfig();
            if (modelType == null && files.Has(Path("config.json"))) {
                modelType = (string)JObject.Parse(await files.ReadTextAsync(Path("config.json")))["model_type"];
            }
            string tokenizerJson = files.Has(Path("tokenizer.json")) ? await files.ReadTextAsync(Path("tokenizer.json")) : null;

            string className = ResolveClass(config.TokenizerClass, modelType, tokenizerJson);
            if (tokenizerJson != null) {
                return className switch {
                    "CLIPTokenizer" => CLIPTokenizer.FromTokenizerJson(tokenizerJson, config),
                    "GPT2Tokenizer" => GPT2Tokenizer.FromTokenizerJson(tokenizerJson, config),
                    "LlamaTokenizer" => LlamaTokenizer.FromTokenizerJson(tokenizerJson, config),
                    "Qwen2Tokenizer" => Qwen2Tokenizer.FromTokenizerJson(tokenizerJson, config),
                    _ => new TokenizersBackend(tokenizerJson, config),
                };
            }
            switch (className) {
                case "CLIPTokenizer":
                    return new CLIPTokenizer(await files.ReadTextAsync(Path("vocab.json")), await files.ReadTextAsync(Path("merges.txt")), config);
                case "GPT2Tokenizer":
                    return new GPT2Tokenizer(await files.ReadTextAsync(Path("vocab.json")), await files.ReadTextAsync(Path("merges.txt")), config);
                case "Qwen2Tokenizer":
                    return new Qwen2Tokenizer(await files.ReadTextAsync(Path("vocab.json")), await files.ReadTextAsync(Path("merges.txt")), config);
                case "LlamaTokenizer" when files.Has(Path("tokenizer.model")):
                    return LlamaTokenizer.FromSentencePiece(await files.ReadFileAsync(Path("tokenizer.model")), config);
            }
            throw new NotSupportedException($"The tokenizer '{config.TokenizerClass}' of '{files.ModelId}' has no tokenizer.json, " +
                "and this package cannot build it from its other files.");
        }

        /// <summary>The class name transformers would load, without "Fast", or null to load tokenizer.json as it is.</summary>
        internal static string ResolveClass(string hubClass, string modelType, string tokenizerJson) {
            hubClass = hubClass?.EndsWith("Fast") == true ? hubClass[..^4] : hubClass;
            bool incorrectHubClass = modelType != null && ModelsWithIncorrectHubTokenizerClass.Contains(modelType);
            string registered = null;
            if (modelType != null && !TokenizerMapping.TryGetValue(modelType, out registered) && incorrectHubClass) {
                registered = "TokenizersBackend";
            }

            // A LlamaTokenizer class with a byte-level tokenizer.json is a mistake of the hub files.
            if (hubClass == "LlamaTokenizer" && !incorrectHubClass && tokenizerJson != null && HasByteLevel(tokenizerJson)) {
                return null;
            }
            if (hubClass != null && registered != null && registered != hubClass) {
                return registered == "TokenizersBackend" ? null : incorrectHubClass ? registered : hubClass;
            }
            return hubClass ?? registered;
        }

        /// <summary>Whether tokenizer.json has a ByteLevel part, which transformers checks on the text of the file.</summary>
        private static bool HasByteLevel(string tokenizerJson) => tokenizerJson.Contains("\"ByteLevel\"");
    }
}
