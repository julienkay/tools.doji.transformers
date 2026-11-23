using Doji.AI.ModelHub;
using Doji.Downloads;
using Newtonsoft.Json;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// One model repository, read through AI Model Hub: its configs, tokenizer files and models.
    /// One code path serves the async and the sync API: in sync mode every
    /// task it returns is already complete.
    /// </summary>
    internal sealed class ModelRepository {

        private readonly PretrainedRepository _repo;
        private readonly bool _sync;
        private readonly IProgress<DownloadProgress> _progress;
        private readonly CancellationToken _cancellationToken;

        public string ModelId { get; }

        private ModelRepository(string modelId, PretrainedRepository repo, bool sync, IProgress<DownloadProgress> progress, CancellationToken cancellationToken) {
            ModelId = modelId;
            _repo = repo;
            _sync = sync;
            _progress = progress;
            _cancellationToken = cancellationToken;
        }

        /// <summary>
        /// Opens a model for loading without waiting. Works in the Editor and for bundled
        /// or already downloaded models on Windows, macOS, Linux, and iOS.
        /// </summary>
        public static ModelRepository Open(string modelId) {
            if (modelId == null) {
                throw new ArgumentNullException(nameof(modelId));
            }
            try {
                return new ModelRepository(modelId, Hub.FromPretrained(modelId), true, null, default);
            } catch (FileNotFoundException e) {
                throw NotDeclared(modelId, e);
            }
        }

        /// <summary>
        /// Opens a model for loading. Works on every platform.
        /// </summary>
        public static async Task<ModelRepository> OpenAsync(string modelId, IProgress<DownloadProgress> progress, CancellationToken cancellationToken) {
            if (modelId == null) {
                throw new ArgumentNullException(nameof(modelId));
            }
            try {
                return new ModelRepository(modelId, await Hub.FromPretrainedAsync(modelId), false, progress, cancellationToken);
            } catch (FileNotFoundException e) {
                throw NotDeclared(modelId, e);
            }
        }

        /// <summary>
        /// Whether the project includes this file of the model.
        /// </summary>
        public bool Has(string repoPath) => _repo.HasFile(repoPath);

        public Task<string> ReadTextAsync(string repoPath) {
            return _sync
                ? Task.FromResult(_repo.ReadText(repoPath))
                : _repo.ReadTextAsync(repoPath, _progress, _cancellationToken);
        }

        public Task<byte[]> ReadFileAsync(string repoPath) {
            return _sync
                ? Task.FromResult(_repo.ReadFile(repoPath))
                : _repo.ReadFileAsync(repoPath, _progress, _cancellationToken);
        }

        public async Task<T> ReadJsonAsync<T>(string repoPath) {
            return JsonConvert.DeserializeObject<T>(await ReadTextAsync(repoPath));
        }

        /// <summary>
        /// Reads a json file the model may not have, for example <c>generation_config.json</c>,
        /// which some models leave out. Returns the default value when it is missing.
        /// </summary>
        public async Task<T> ReadOptionalJsonAsync<T>(string repoPath) {
            return Has(repoPath) ? await ReadJsonAsync<T>(repoPath) : default;
        }

        public Task<Model> LoadModelAsync(string repoPath) {
            _cancellationToken.ThrowIfCancellationRequested();
            return _sync
                ? Task.FromResult(_repo.LoadModel(repoPath))
                : _repo.LoadModelAsync(repoPath, _progress, _cancellationToken);
        }

        /// <summary>
        /// Opens a model whose weights stay in its file until a worker needs them, so they are not in RAM.
        /// </summary>
        public Task<ModelFile> OpenModelAsync(string repoPath) {
            _cancellationToken.ThrowIfCancellationRequested();
            return _sync
                ? Task.FromResult(_repo.OpenModel(repoPath))
                : _repo.OpenModelAsync(repoPath, _progress, _cancellationToken);
        }

        private static Exception NotDeclared(string modelId, Exception inner) {
            return new FileNotFoundException(
                $"tools.doji.transformers could not find the model '{modelId}'. " +
                $"Add it to a model-hub-manifest.json (for example the project manifest in Project Settings > Doji > AI > AI Model Hub) " +
                $"and download it in Window > AI Model Hub. Review the model's license at https://huggingface.co/{modelId}. {inner.Message}",
                inner);
        }
    }
}
