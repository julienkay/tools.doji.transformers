using System;
using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Gets the tokens of <see cref="PreTrainedModel.Generate"/> as they come: first the prompt, then each new token.
    /// </summary>
    public abstract class BaseStreamer {

        public abstract void Put(IReadOnlyList<int> tokenIds);

        /// <summary>
        /// Called when generation ends.
        /// </summary>
        public abstract void End();
    }

    /// <summary>
    /// Decodes the tokens as they come and passes on each new piece of text, as transformers' <c>TextStreamer</c> does.
    /// </summary>
    public class TextStreamer : BaseStreamer {

        private readonly PreTrainedTokenizerBase _tokenizer;
        private readonly Action<string> _onText;
        private readonly bool _skipSpecialTokens;
        private bool _skipPrompt;
        private readonly List<int> _tokens = new();
        private int _printed;

        /// <param name="onText">Gets each new piece of text.</param>
        /// <param name="skipPrompt">Whether to leave out the prompt.</param>
        /// <param name="skipSpecialTokens">Whether to leave out special tokens such as the end-of-sequence token.</param>
        public TextStreamer(PreTrainedTokenizerBase tokenizer, Action<string> onText, bool skipPrompt = false, bool skipSpecialTokens = true) {
            _tokenizer = tokenizer;
            _onText = onText;
            _skipPrompt = skipPrompt;
            _skipSpecialTokens = skipSpecialTokens;
        }

        public override void Put(IReadOnlyList<int> tokenIds) {
            if (_skipPrompt) {
                _skipPrompt = false;
                return;
            }
            _tokens.AddRange(tokenIds);
            string text = _tokenizer.Decode(_tokens, skipSpecialTokens: _skipSpecialTokens);
            // Wait for the rest of a character that spans several tokens.
            if (!text.EndsWith("�") && text.Length > _printed) {
                _onText(text[_printed..]);
                _printed = text.Length;
            }
        }

        public override void End() {
            string text = _tokenizer.Decode(_tokens, skipSpecialTokens: _skipSpecialTokens);
            if (text.Length > _printed) {
                _onText(text[_printed..]);
            }
            _tokens.Clear();
            _printed = 0;
        }
    }
}
