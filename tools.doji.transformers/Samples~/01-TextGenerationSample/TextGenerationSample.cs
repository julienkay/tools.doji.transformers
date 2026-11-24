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

        /// <summary>
        /// The models to choose from. Each must be listed in an AI Model Hub manifest. SmolLM2-1.7B and Phi-3-mini
        /// are optional there: turn them on in AI Model Hub to download them.
        /// </summary>
        public string[] ModelIds = {
            "julienkay/SmolLM2-360M-Instruct-onnx",
            "julienkay/SmolLM2-1.7B-Instruct-onnx",
            "julienkay/SmolLM2-135M-Instruct-onnx",
            "julienkay/Phi-3-mini-4k-instruct-onnx",
        };
        public int MaxNewTokens = 256;

        /// <summary>
        /// Sampling settings, as the model card of SmolLM2 recommends. Greedy decoding (DoSample off) makes small
        /// models repeat earlier answers of the chat.
        /// </summary>
        public bool DoSample = true;
        public float Temperature = 0.2f;
        public float TopP = 0.9f;

        /// <summary>The prompts that the empty chat offers.</summary>
        public string[] Suggestions = {
            "What is the capital of France?",
            "Tell me a joke.",
            "Explain gravity in one sentence.",
            "Write a haiku about the sea.",
        };

        private PreTrainedModel _model;
        private PreTrainedTokenizerBase _tokenizer;
        private readonly List<ChatMessage> _messages = new();
        private CancellationTokenSource _cancel = new();
        private int _selected = -1;
        private bool _loading, _generating;

        /// <summary>Whether the chat follows new text. Scrolling up stops it, scrolling to the end starts it again.</summary>
        private bool _stickToEnd = true;

        /// <summary>Counts the cleared chats, so a generation that ends after a clear leaves the new chat alone.</summary>
        private int _chat;

        private DropdownField _modelField;
        private ScrollView _messageList;
        private VisualElement _empty, _suggestions, _composer;
        private Label _emptyHint, _status;
        private TextField _input;
        private Button _send, _clear;
        private ChatIcon _sendIcon;

        private void OnEnable() {
            var root = GetComponent<UIDocument>().rootVisualElement;
            _modelField = root.Q<DropdownField>("model");
            _messageList = root.Q<ScrollView>("messages");
            _empty = root.Q("empty");
            _emptyHint = root.Q<Label>("emptyHint");
            _suggestions = root.Q("suggestions");
            _status = root.Q<Label>("status");
            _input = root.Q<TextField>("input");
            _composer = _input.parent;
            _send = root.Q<Button>("send");
            _sendIcon = root.Q<ChatIcon>("sendIcon");
            _clear = root.Q<Button>("clear");

            _messageList.contentContainer.RegisterCallback<GeometryChangedEvent>(_ => {
                if (_stickToEnd) {
                    _messageList.scrollOffset = new Vector2(0, float.MaxValue);
                }
            });
            _messageList.verticalScroller.valueChanged += value =>
                _stickToEnd = _messageList.verticalScroller.highValue - value < 40f;

            _modelField.choices = ModelIds.Select(DisplayName).ToList();
            _modelField.RegisterValueChangedCallback(e => Load(_modelField.index));
            _modelField.Q(className: "unity-base-popup-field__arrow")?.Add(new ChatIcon(ChatIcon.Shape.ChevronDown));

            _input.textEdition.placeholder = "Message the model";
            _input.RegisterCallback<KeyDownEvent>(OnInputKey, TrickleDown.TrickleDown);
            _input.RegisterValueChangedCallback(_ => UpdateControls());
            _input.RegisterCallback<FocusInEvent>(_ => _composer.AddToClassList("composer--focused"));
            _input.RegisterCallback<FocusOutEvent>(_ => _composer.RemoveFromClassList("composer--focused"));
            _send.clicked += OnSendOrStop;
            _clear.clicked += Clear;

            _suggestions.Clear();
            foreach (string suggestion in Suggestions) {
                var chip = new Button(() => {
                    if (CanSendText(suggestion)) {
                        Send(suggestion);
                    }
                }) { text = suggestion };
                chip.AddToClassList("suggestion");
                _suggestions.Add(chip);
            }
            UpdateControls();
        }

        private void Start() {
            _modelField.index = 0;
        }

        private static string DisplayName(string modelId) {
            string name = modelId.Split('/').Last();
            return name.EndsWith("-onnx") ? name[..^5] : name;
        }

        private void OnInputKey(KeyDownEvent evt) {
            // Enter sends, Shift+Enter adds a line.
            bool enter = evt.keyCode is KeyCode.Return or KeyCode.KeypadEnter || evt.character == '\n';
            if (enter && !evt.shiftKey) {
                evt.StopImmediatePropagation();
                if (evt.keyCode != KeyCode.None && CanSend) {
                    Send(_input.value);
                }
            }
        }

        private bool CanSendText(string text) => _model != null && !_loading && !_generating && !string.IsNullOrWhiteSpace(text);
        private bool CanSend => CanSendText(_input.value);

        private void OnSendOrStop() {
            if (_generating) {
                Stop();
            } else if (CanSend) {
                Send(_input.value);
            }
        }

        private async void Load(int index) {
            Unload();
            _selected = index;
            string modelId = ModelIds[index];
            _loading = true;
            UpdateControls();
            SetStatus($"Loading {DisplayName(modelId)}...");
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
                SetStatus(ReadyText);
            } catch (OperationCanceledException) {
                return;
            } catch (Exception e) {
                Debug.LogException(e);
                SetStatus($"Could not load {DisplayName(modelId)}. Check that AI Model Hub downloaded it, and see the Console.");
            }
            _loading = false;
            UpdateControls();
            _input.Focus();
        }

        private string ReadyText => "Small language models make mistakes. Check important facts.";

        private void Unload() {
            Stop();
            _model?.Dispose();
            _model = null;
            _tokenizer = null;
            ClearMessages();
        }

        private async void Send(string text) {
            text = text.Trim();
            _input.value = "";
            _messages.Add(ChatMessage.User(text));
            AddUserMessage(text);
            Label reply = AddAssistantMessage();
            VisualElement message = reply.parent.parent;
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
                    message.RemoveFromClassList("message--pending");
                    reply.text = replyText.Trim();
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
                message.RemoveFromClassList("message--pending");
                message.AddToClassList("message--error");
                reply.text = "Could not generate an answer. See the Console.";
                SetStatus("");
            }
            if (chat == _chat) {
                // Keep what the model wrote, also when it was stopped, so the chat goes on from there.
                if (replyText.Trim().Length > 0) {
                    _messages.Add(ChatMessage.Assistant(replyText.Trim()));
                } else if (message.ClassListContains("message--pending")) {
                    message.RemoveFromHierarchy();
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
            SetStatus(_model != null ? ReadyText : "");
            UpdateControls();
            _input.Focus();
        }

        private void ClearMessages() {
            _chat++;
            _messages.Clear();
            _messageList?.Clear();
        }

        private void AddUserMessage(string text) {
            var message = NewMessage("message--user");
            var bubble = new VisualElement();
            bubble.AddToClassList("message__bubble");
            bubble.Add(NewText(text));
            message.Add(bubble);
            AddMessage(message);
        }

        /// <summary>Adds an empty answer that shows a typing indicator until the first token, and returns its text.</summary>
        private Label AddAssistantMessage() {
            var message = NewMessage("message--assistant", "message--pending");
            var avatar = new VisualElement();
            avatar.AddToClassList("message__avatar");
            var avatarIcon = new ChatIcon(ChatIcon.Shape.Sparkle);
            avatarIcon.AddToClassList("message__avatar-icon");
            avatar.Add(avatarIcon);
            message.Add(avatar);

            var body = new VisualElement();
            body.AddToClassList("message__body");
            var typing = new VisualElement();
            typing.AddToClassList("typing");
            for (int i = 0; i < 3; i++) {
                var dot = new VisualElement();
                dot.AddToClassList("typing__dot");
                typing.Add(dot);
            }
            int phase = 0;
            typing.schedule.Execute(() => {
                if (!message.ClassListContains("message--pending")) {
                    typing.RemoveFromHierarchy();
                    return;
                }
                for (int i = 0; i < typing.childCount; i++) {
                    typing[i].EnableInClassList("typing__dot--on", i == phase % 3);
                }
                phase++;
            }).Every(300);
            body.Add(typing);
            var text = NewText("");
            body.Add(text);
            message.Add(body);
            AddMessage(message);
            return text;
        }

        private static VisualElement NewMessage(params string[] classes) {
            var message = new VisualElement();
            message.AddToClassList("message");
            message.AddToClassList("message--enter");
            foreach (string c in classes) {
                message.AddToClassList(c);
            }
            // Fades and slides in once it has a layout.
            message.RegisterCallbackOnce<GeometryChangedEvent>(_ => message.RemoveFromClassList("message--enter"));
            return message;
        }

        private static Label NewText(string text) {
            var label = new Label(text) { enableRichText = false };
            label.selection.isSelectable = true;
            label.AddToClassList("message__text");
            return label;
        }

        private void AddMessage(VisualElement message) {
            _messageList.Add(message);
            UpdateEmpty();
            // A new message always scrolls to it. The layout scrolls once it knows the new height.
            _stickToEnd = true;
        }

        private void SetStatus(string text) {
            if (_status != null) {
                _status.text = text;
            }
        }

        private void UpdateEmpty() {
            _empty?.EnableInClassList("hidden", _messageList.childCount > 0);
        }

        private void UpdateControls() {
            if (_send == null) {
                return;
            }
            UpdateEmpty();
            _emptyHint.text = _loading
                ? "Loading the model..."
                : _model != null ? $"{DisplayName(ModelIds[_selected])} runs on this device." : "";
            _modelField.SetEnabled(!_generating);
            _input.SetEnabled(_model != null && !_loading);
            _sendIcon.IconShape = _generating ? ChatIcon.Shape.Stop : ChatIcon.Shape.Send;
            _send.EnableInClassList("send-button--stop", _generating);
            _send.SetEnabled(_generating || CanSend);
            _clear.SetEnabled(_model != null && !_loading);
            _suggestions.SetEnabled(_model != null && !_loading && !_generating);
        }

        private void OnDestroy() {
            Unload();
        }
    }
}
