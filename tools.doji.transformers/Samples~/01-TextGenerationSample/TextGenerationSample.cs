using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;
using UnityEngine.UIElements;

namespace Doji.AI.Transformers.Samples.TextGeneration {

    /// <summary>
    /// A chat with a small language model. The models come through AI Model Hub: the
    /// <c>TextGenerationModels</c> asset next to this script lists them, and Model Hub downloads them.
    /// The UI is a UI Toolkit document, <c>UI/TextGenerationSample.uxml</c>.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
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
        private bool _loading, _generating;

        /// <summary>Counts the cleared chats, so a generation that ends after a clear leaves the new chat alone.</summary>
        private int _chat;

        private DropdownField _modelField;
        private ScrollView _messageList;
        private Label _empty, _status;
        private TextField _input;
        private Button _send, _clear;

        private void OnEnable() {
            var root = GetComponent<UIDocument>().rootVisualElement;
            _modelField = root.Q<DropdownField>("model");
            _messageList = root.Q<ScrollView>("messages");
            _empty = root.Q<Label>("empty");
            _status = root.Q<Label>("status");
            _input = root.Q<TextField>("input");
            _send = root.Q<Button>("send");
            _clear = root.Q<Button>("clear");

            _modelField.choices = ModelIds.Select(id => id.Split('/').Last()).ToList();
            _modelField.RegisterValueChangedCallback(e => Load(_modelField.index));
            _input.textEdition.placeholder = "Message the model (Enter sends, Shift+Enter adds a line)";
            _input.RegisterCallback<KeyDownEvent>(OnInputKey, TrickleDown.TrickleDown);
            _send.clicked += OnSendOrStop;
            _clear.clicked += Clear;
            UpdateControls();
        }

        private void Start() {
            _modelField.index = 0;
        }

        private void OnInputKey(KeyDownEvent evt) {
            bool enter = evt.keyCode is KeyCode.Return or KeyCode.KeypadEnter || evt.character == '\n';
            if (enter && !evt.shiftKey) {
                evt.StopImmediatePropagation();
                if (evt.keyCode != KeyCode.None && CanSend) {
                    Send();
                }
            }
        }

        private bool CanSend => _model != null && !_loading && !_generating && !string.IsNullOrWhiteSpace(_input.value);

        private void OnSendOrStop() {
            if (_generating) {
                Stop();
            } else if (CanSend) {
                Send();
            }
        }

        private async void Load(int index) {
            Unload();
            _selected = index;
            string modelId = ModelIds[index];
            _loading = true;
            SetStatus($"Loading {modelId}...");
            var cancel = _cancel;
            try {
                var tokenizer = await AutoTokenizer.FromPretrainedAsync(modelId, cancellationToken: cancel.Token);
                var model = await AutoModelForCausalLM.FromPretrainedAsync(modelId, cancellationToken: cancel.Token);
                if (cancel.IsCancellationRequested) {
                    model.Dispose();
                    return;
                }
                _tokenizer = tokenizer;
                _model = model;
                SetStatus($"{modelId} is ready.");
            } catch (OperationCanceledException) {
                return;
            } catch (Exception e) {
                Debug.LogException(e);
                SetStatus($"Could not load {modelId}. See the Console.");
            }
            _loading = false;
            UpdateControls();
            _input.Focus();
        }

        private void Unload() {
            Stop();
            _model?.Dispose();
            _model = null;
            _tokenizer = null;
            ClearMessages();
        }

        private async void Send() {
            string text = _input.value.Trim();
            _input.value = "";
            _messages.Add(ChatMessage.User(text));
            AddBubble(text, "message--user");
            Label reply = AddBubble("Thinking...", "message--assistant", "message--pending");
            var bubble = reply.parent;
            _generating = true;
            UpdateControls();

            var cancel = _cancel;
            int chat = _chat;
            string replyText = "";
            try {
                string prompt = _tokenizer.ApplyChatTemplate(_messages);
                int[] promptIds = _tokenizer.Encode(prompt, addSpecialTokens: false).InputIds.ToArray();
                var streamer = new TextStreamer(_tokenizer, piece => {
                    replyText += piece;
                    bubble.RemoveFromClassList("message--pending");
                    reply.text = replyText.TrimStart();
                    ScrollToEnd();
                }, skipPrompt: true);
                var config = _model.GenerationConfig.Clone();
                config.MaxNewTokens = MaxNewTokens;
                config.DoSample = DoSample;
                config.Temperature = Temperature;
                config.TopP = TopP;
                float start = Time.realtimeSinceStartup;
                int[] sequence = await _model.GenerateAsync(promptIds, config, streamer, cancellationToken: cancel.Token);
                int newTokens = sequence.Length - promptIds.Length;
                SetStatus($"{newTokens} tokens at {newTokens / (Time.realtimeSinceStartup - start):F1} tokens/s");
            } catch (OperationCanceledException) {
                SetStatus("Stopped.");
            } catch (Exception e) {
                Debug.LogException(e);
                bubble.AddToClassList("message--error");
                SetStatus("Could not generate. See the Console.");
            }
            if (chat == _chat) {
                // Keep what the model wrote, also when it was stopped, so the chat goes on from there.
                if (replyText.Trim().Length > 0) {
                    _messages.Add(ChatMessage.Assistant(replyText.Trim()));
                } else if (bubble.ClassListContains("message--pending")) {
                    bubble.RemoveFromHierarchy();
                }
            }
            _generating = false;
            UpdateControls();
            _input.Focus();
        }

        /// <summary>
        /// Stops the generation. It ends at the next token: until then, <see cref="_generating"/> stays true, so
        /// no other generation starts on the same model.
        /// </summary>
        private void Stop() {
            _cancel.Cancel();
            _cancel = new CancellationTokenSource();
        }

        private void Clear() {
            Stop();
            ClearMessages();
            SetStatus(_model != null ? $"{ModelIds[_selected]} is ready." : "");
            UpdateControls();
        }

        private void ClearMessages() {
            _chat++;
            _messages.Clear();
            _messageList?.Clear();
            if (_empty != null) {
                _messageList.Add(_empty);
            }
        }

        /// <summary>Adds a chat bubble and returns its text. The bubble is the parent of the text.</summary>
        private Label AddBubble(string text, params string[] classes) {
            _empty.RemoveFromHierarchy();
            // The padding is on the bubble, not on the label: a label with padding and a max-width wraps its
            // text wider than it measures it, which clips the last line.
            var bubble = new VisualElement();
            bubble.AddToClassList("message");
            foreach (string c in classes) {
                bubble.AddToClassList(c);
            }
            var label = new Label(text) { enableRichText = false };
            label.selection.isSelectable = true;
            label.AddToClassList("message__text");
            bubble.Add(label);
            _messageList.Add(bubble);
            ScrollToEnd();
            return label;
        }

        private void ScrollToEnd() {
            // After the layout, so the scroller knows the new height.
            _messageList.schedule.Execute(() => _messageList.scrollOffset = new Vector2(0, float.MaxValue));
        }

        private void SetStatus(string text) {
            if (_status != null) {
                _status.text = text;
            }
        }

        private void UpdateControls() {
            if (_send == null) {
                return;
            }
            _modelField.SetEnabled(!_generating);
            _input.SetEnabled(_model != null && !_loading);
            _send.text = _generating ? "Stop" : "Send";
            _send.EnableInClassList("button--stop", _generating);
            _send.SetEnabled(_generating || (_model != null && !_loading));
            _clear.SetEnabled(_model != null && !_loading);
        }

        private void OnDestroy() {
            Unload();
        }
    }
}
