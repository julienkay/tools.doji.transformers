using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

namespace Doji.AI.Transformers.Samples.TextGeneration {

    /// <summary>
    /// A simple chat with a small language model. The models come through AI Model Hub: the
    /// <c>TextGenerationModels</c> asset next to this script lists them, and Model Hub downloads them.
    /// </summary>
    public class TextGenerationSample : MonoBehaviour {

        /// <summary>The models to choose from. Each must be listed in an AI Model Hub manifest.</summary>
        public string[] ModelIds = {
            "julienkay/SmolLM2-135M-Instruct-onnx",
        };
        public int MaxNewTokens = 256;

        /// <summary>
        /// Sampling settings, as the model card of SmolLM2 recommends. Greedy decoding (DoSample off) makes small
        /// models repeat earlier answers of the chat.
        /// </summary>
        public bool DoSample = true;
        public float Temperature = 0.2f;
        public float TopP = 0.9f;

        private PreTrainedModel _model;
        private PreTrainedTokenizerBase _tokenizer;
        private readonly List<ChatMessage> _messages = new();
        private CancellationTokenSource _cancel = new();
        private int _selected = -1;
        private string _input = "What is the capital of France?";
        private string _reply = "";
        private string _status = "";
        private bool _busy;
        private Vector2 _scroll;
        private GUIStyle _messageStyle;

        private void Start() {
            Load(0);
        }

        private async void Load(int index) {
            Unload();
            _selected = index;
            string modelId = ModelIds[index];
            _busy = true;
            var cancel = _cancel;
            try {
                _status = $"Loading {modelId}...";
                var tokenizer = await AutoTokenizer.FromPretrainedAsync(modelId, cancellationToken: cancel.Token);
                var model = await AutoModelForCausalLM.FromPretrainedAsync(modelId, cancellationToken: cancel.Token);
                if (cancel.IsCancellationRequested) {
                    model.Dispose();
                    return;
                }
                _tokenizer = tokenizer;
                _model = model;
                _status = modelId;
            } catch (OperationCanceledException) {
                return;
            } catch (Exception e) {
                Debug.LogException(e);
                _status = $"Could not load {modelId}. See the Console.";
            }
            _busy = false;
        }

        private void Unload() {
            _cancel.Cancel();
            _cancel = new CancellationTokenSource();
            _model?.Dispose();
            _model = null;
            _tokenizer = null;
            _messages.Clear();
            _reply = "";
            _busy = false;
        }

        private async void Send() {
            _busy = true;
            _messages.Add(ChatMessage.User(_input));
            _input = "";
            _reply = "";
            var cancel = _cancel;
            try {
                string prompt = _tokenizer.ApplyChatTemplate(_messages);
                int[] promptIds = _tokenizer.Encode(prompt, addSpecialTokens: false).InputIds.ToArray();
                var streamer = new TextStreamer(_tokenizer, text => _reply += text, skipPrompt: true);
                var config = _model.GenerationConfig.Clone();
                config.MaxNewTokens = MaxNewTokens;
                config.DoSample = DoSample;
                config.Temperature = Temperature;
                config.TopP = TopP;
                float start = Time.realtimeSinceStartup;
                int[] sequence = await _model.GenerateAsync(promptIds, config, streamer, cancellationToken: cancel.Token);
                int newTokens = sequence.Length - promptIds.Length;
                _status = $"{ModelIds[_selected]}: {newTokens} tokens, {newTokens / (Time.realtimeSinceStartup - start):F1} tokens/s";
                _messages.Add(ChatMessage.Assistant(_reply.Trim()));
                _reply = "";
            } catch (OperationCanceledException) {
                return;
            } catch (Exception e) {
                Debug.LogException(e);
                _status = "Could not generate. See the Console.";
            }
            _busy = false;
        }

        private void OnGUI() {
            _messageStyle ??= new GUIStyle(GUI.skin.label) { richText = true, wordWrap = true };
            GUILayout.BeginArea(new Rect(10, 10, Screen.width - 20, Screen.height - 20));

            GUI.enabled = !_busy;
            int selected = GUILayout.Toolbar(_selected, ModelIds.Select(id => id.Split('/').Last()).ToArray());
            GUI.enabled = true;
            if (selected != _selected) {
                Load(selected);
            }
            GUILayout.Label(_status);

            _scroll = GUILayout.BeginScrollView(_scroll);
            foreach (var message in _messages) {
                GUILayout.Label($"<b>{message.Role}</b>: {message.Content}", _messageStyle);
            }
            if (_reply.Length > 0) {
                GUILayout.Label($"<b>assistant</b>: {_reply}", _messageStyle);
            }
            GUILayout.EndScrollView();

            GUI.enabled = !_busy && _model != null;
            GUILayout.BeginHorizontal();
            _input = GUILayout.TextField(_input);
            if (GUILayout.Button("Send", GUILayout.Width(80)) && _input.Length > 0) {
                Send();
            }
            GUI.enabled = _model != null && _messages.Count > 0;
            if (GUILayout.Button("Clear", GUILayout.Width(80))) {
                _cancel.Cancel();
                _cancel = new CancellationTokenSource();
                _messages.Clear();
                _reply = "";
                _busy = false;
            }
            GUILayout.EndHorizontal();
            GUI.enabled = true;
            GUILayout.EndArea();
        }

        private void OnDestroy() {
            Unload();
        }
    }
}
