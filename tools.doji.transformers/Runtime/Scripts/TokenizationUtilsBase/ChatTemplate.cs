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
        /// In the ChatML form of Qwen3.5, the generation prompt opens a thinking block, which is empty unless
        /// <paramref name="enableThinking"/> is true.
        /// </remarks>
        /// <param name="addGenerationPrompt">Whether to end with the start of an assistant message, for generation.</param>
        /// <param name="enableThinking">The <c>enable_thinking</c> argument of the template: whether the model may
        /// reason in a thinking block before it answers.</param>
        public string ApplyChatTemplate(IEnumerable<ChatMessage> messages, bool addGenerationPrompt = true, bool enableThinking = false) {
            string template = Config.ChatTemplate?.Type == JTokenType.String ? (string)Config.ChatTemplate : null;
            var list = messages.ToList();
            var text = new StringBuilder();
            if (template != null && template.Contains("<|im_start|>")) {
                if (list.Count > 0 && list[0].Role != "system") {
                    // The line breaks are real ones, or "\n" escapes in the template's string literal (Qwen2.5).
                    Match system = Regex.Match(template, @"'<\|im_start\|>system(?:\n|\\n)([^']*?)<\|im_end\|>(?:\n|\\n)'", RegexOptions.Singleline);
                    if (system.Success) {
                        text.Append($"<|im_start|>system\n{system.Groups[1].Value}<|im_end|>\n");
                    }
                }
                foreach (ChatMessage message in list) {
                    text.Append($"<|im_start|>{message.Role}\n{message.Content}<|im_end|>\n");
                }
                if (addGenerationPrompt) {
                    text.Append("<|im_start|>assistant\n");
                    if (template.Contains("enable_thinking is defined and enable_thinking is true")) {
                        text.Append(enableThinking ? "<think>\n" : "<think>\n\n</think>\n\n");
                    }
                }
            } else if (template != null && template.Contains("<|end|>")) {
                if (template.Contains("bos_token")) {
                    text.Append(BosToken);
                }
                foreach (ChatMessage message in list) {
                    text.Append($"<|{message.Role}|>\n{message.Content}<|end|>\n");
                }
                text.Append(addGenerationPrompt ? "<|assistant|>\n" : EosToken);
            } else {
                throw new NotSupportedException("This chat template is not supported. Supported: ChatML and Phi-3.");
            }
            return text.ToString();
        }
    }
}
