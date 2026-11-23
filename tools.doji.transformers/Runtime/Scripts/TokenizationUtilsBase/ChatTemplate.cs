using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// One message of a chat, for <see cref="PreTrainedTokenizerBase.ApplyChatTemplate"/>.
    /// </summary>
    public readonly struct ChatMessage {

        /// <summary>"system", "user" or "assistant".</summary>
        public string Role { get; }

        public string Content { get; }

        public ChatMessage(string role, string content) {
            Role = role;
            Content = content;
        }

        public static ChatMessage System(string content) => new ChatMessage("system", content);
        public static ChatMessage User(string content) => new ChatMessage("user", content);
        public static ChatMessage Assistant(string content) => new ChatMessage("assistant", content);
    }

    public abstract partial class PreTrainedTokenizerBase {

        /// <summary>
        /// Formats a chat as the model's chat template does.
        /// </summary>
        /// <remarks>
        /// The templates are Jinja, which this package does not run. It recognizes two common forms instead:
        /// ChatML (<c>&lt;|im_start|&gt;role\ncontent&lt;|im_end|&gt;\n</c>, SmolLM2 and Qwen), with the default system
        /// prompt the template adds when the chat has none, and Phi-3 (<c>&lt;|role|&gt;\ncontent&lt;|end|&gt;\n</c>).
        /// </remarks>
        /// <param name="addGenerationPrompt">Whether to end with the start of an assistant message, for generation.</param>
        public string ApplyChatTemplate(IEnumerable<ChatMessage> messages, bool addGenerationPrompt = true) {
            string template = Config.ChatTemplate?.Type == JTokenType.String ? (string)Config.ChatTemplate : null;
            var list = messages.ToList();
            var text = new StringBuilder();
            if (template != null && template.Contains("<|im_start|>")) {
                if (list.Count > 0 && list[0].Role != "system") {
                    Match system = Regex.Match(template, @"'<\|im_start\|>system\n(.*?)<\|im_end\|>\n'", RegexOptions.Singleline);
                    if (system.Success) {
                        text.Append($"<|im_start|>system\n{system.Groups[1].Value}<|im_end|>\n");
                    }
                }
                foreach (ChatMessage message in list) {
                    text.Append($"<|im_start|>{message.Role}\n{message.Content}<|im_end|>\n");
                }
                if (addGenerationPrompt) {
                    text.Append("<|im_start|>assistant\n");
                }
            } else if (template != null && template.Contains("<|end|>")) {
                if (template.Contains("bos_token")) {
                    text.Append(BosToken?.Content);
                }
                foreach (ChatMessage message in list) {
                    text.Append($"<|{message.Role}|>\n{message.Content}<|end|>\n");
                }
                text.Append(addGenerationPrompt ? "<|assistant|>\n" : EosToken?.Content);
            } else {
                throw new NotSupportedException("This chat template is not supported. Supported: ChatML and Phi-3.");
            }
            return text.ToString();
        }
    }
}
