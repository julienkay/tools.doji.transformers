using Doji.Downloads;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Loads the tokenizer of a pretrained model, choosing the class from its <c>tokenizer_config.json</c>.
    /// </summary>
    public static class AutoTokenizer {

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
        /// Like transformers, prefers the fast tokenizer of <c>tokenizer.json</c>, except for sentencepiece models.
        /// </summary>
        internal static async Task<PreTrainedTokenizerBase> FromPretrainedAsync(ModelRepository files, string subfolder = "") {
            string Path(string name) => subfolder + name;
            var config = TokenizerConfig.Deserialize(await files.ReadTextAsync(Path(PreTrainedTokenizerBase.TOKENIZER_CONFIG_FILE)));
            string tokenizerClass = config.TokenizerClass ?? "";
            if (tokenizerClass.StartsWith("CLIPTokenizer")) {
                var vocab = await files.ReadJsonAsync<Vocab>(Path("vocab.json"));
                return new ClipTokenizer(vocab, await files.ReadTextAsync(Path("merges.txt")), config);
            }
            if (tokenizerClass.StartsWith("LlamaTokenizer") && files.Has(Path("tokenizer.model"))) {
                byte[] model = await files.ReadFileAsync(Path("tokenizer.model"));
                return new LlamaTokenizer(WriteTemporaryFile(files.ModelId, model), config);
            }
            if (files.Has(Path("tokenizer.json"))) {
                return new PreTrainedTokenizerFast(await files.ReadTextAsync(Path("tokenizer.json")), config);
            }
            if (tokenizerClass.StartsWith("GPT2Tokenizer")) {
                var vocab = await files.ReadJsonAsync<Vocab>(Path("vocab.json"));
                return new GPT2Tokenizer(vocab, await files.ReadTextAsync(Path("merges.txt")), config);
            }
            throw new NotSupportedException($"The tokenizer '{tokenizerClass}' of '{files.ModelId}' is not supported.");
        }

        /// <summary>
        /// The sentencepiece library reads its model from a file.
        /// </summary>
        private static string WriteTemporaryFile(string modelId, byte[] data) {
            string folder = System.IO.Path.Combine(Application.temporaryCachePath, modelId);
            Directory.CreateDirectory(folder);
            string path = System.IO.Path.Combine(folder, "tokenizer.model");
            File.WriteAllBytes(path, data);
            return path;
        }
    }
}
