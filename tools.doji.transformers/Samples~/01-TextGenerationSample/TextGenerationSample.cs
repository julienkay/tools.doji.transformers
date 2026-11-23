using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

namespace Doji.AI.Transformers.Samples.TextGeneration {

    /// <summary>
    /// A simple chat with a small language model. The model comes through AI Model Hub:
    /// list it in Project Settings > Doji > AI > AI Model Hub and download it first.
    /// </summary>
    public class TextGenerationSample : MonoBehaviour {

        public string ModelId = "julienkay/SmolLM2-135M-Instruct-onnx";
        public int MaxNewTokens = 256;

        private PreTrainedModel _model;
        private PreTrainedTokenizerBase _tokenizer;
        private readonly List<ChatMessage> _messages = new();
        private readonly CancellationTokenSource _cancel = new();
        private string _input = "What is the capital of France?";
        private string _reply = "";
        private string _status = "";
        private bool _busy = true;
        private Vector2 _scroll;

        private async void Start() {
            try {
                _status = $"Loading {ModelId}...";
                _tokenizer = await AutoTokenizer.FromPretrainedAsync(ModelId, cancellationToken: _cancel.Token);
                _model = await AutoModelForCausalLM.FromPretrainedAsync(ModelId, cancellationToken: _cancel.Token);
                _status = ModelId;
                _busy = false;
            } catch (OperationCanceledException) {
            } catch (Exception e) {
                Debug.LogException(e);
                _status = $"Could not load {ModelId}. See the Console.";
            }
        }

        private async void Send() {
            _busy = true;
            _messages.Add(ChatMessage.User(_input));
            _input = "";
            _reply = "";
            try {
                string prompt = _tokenizer.ApplyChatTemplate(_messages);
                int[] promptIds = _tokenizer.Encode(prompt).InputIds.ToArray();
                var streamer = new TextStreamer(_tokenizer, text => _reply += text, skipPrompt: true);
                var config = _model.GenerationConfig.Clone();
                config.MaxNewTokens = MaxNewTokens;
                float start = Time.realtimeSinceStartup;
                int[] sequence = await _model.GenerateAsync(promptIds, config, streamer, cancellationToken: _cancel.Token);
                int newTokens = sequence.Length - promptIds.Length;
                _status = $"{ModelId}: {newTokens} tokens, {newTokens / (Time.realtimeSinceStartup - start):F1} tokens/s";
                _messages.Add(ChatMessage.Assistant(_reply));
                _reply = "";
            } catch (OperationCanceledException) {
            } catch (Exception e) {
                Debug.LogException(e);
                _status = "Could not generate. See the Console.";
            }
            _busy = false;
        }

        private void OnGUI() {
            GUILayout.BeginArea(new Rect(10, 10, Screen.width - 20, Screen.height - 20));
            GUILayout.Label(_status);
            _scroll = GUILayout.BeginScrollView(_scroll);
            foreach (var message in _messages) {
                GUILayout.Label($"<b>{message.Role}</b>: {message.Content}", new GUIStyle(GUI.skin.label) { richText = true, wordWrap = true });
            }
            if (_reply.Length > 0) {
                GUILayout.Label($"<b>assistant</b>: {_reply}", new GUIStyle(GUI.skin.label) { richText = true, wordWrap = true });
            }
            GUILayout.EndScrollView();
            GUI.enabled = !_busy;
            GUILayout.BeginHorizontal();
            _input = GUILayout.TextField(_input);
            if (GUILayout.Button("Send", GUILayout.Width(80)) && _input.Length > 0) {
                Send();
            }
            GUILayout.EndHorizontal();
            GUI.enabled = true;
            GUILayout.EndArea();
        }

        private void OnDestroy() {
            _cancel.Cancel();
            _model?.Dispose();
        }
    }
}
