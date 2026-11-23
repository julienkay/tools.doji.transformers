using System.Collections.Generic;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The parts of ftfy's <c>fix_text</c> (ftfy 6, default settings) that change ordinary text, which
    /// <see cref="ClipTokenizer"/> runs before tokenizing, as transformers' CLIPTokenizer does when ftfy is installed.
    /// The repair of mojibake (<c>fix_encoding</c>) is left out.
    /// </summary>
    internal static class FixText {

        private static readonly Regex TerminalEscapes = new(@"\x1b\[((?:\d|;)*)([a-zA-Z])");
        private static readonly Regex SingleQuotes = new(@"[\u02bc\u2018-\u201b]");
        private static readonly Regex DoubleQuotes = new(@"[\u201c-\u201f]");
        private static readonly Regex LineBreaks = new(@"\r\n|\r|\u2028|\u2029|\u0085");
        private static readonly Regex ControlChars = new(@"[\u0000-\u0008\u000b\u000e-\u001f\u007f\u206a-\u206f\ufeff\ufff9-\ufffc]");
        /// <summary>The NFKC forms of the ligatures ftfy replaces (Mono's normalization leaves them alone).</summary>
        private static readonly Dictionary<char, string> Ligatures = new() {
            ['\u0132'] = "IJ",
            ['\u0133'] = "ij",
            ['\u0149'] = "\u02bcn",
            ['\u01c4'] = "D\u017d",
            ['\u01c5'] = "D\u017e",
            ['\u01c6'] = "d\u017e",
            ['\u01c7'] = "LJ",
            ['\u01c8'] = "Lj",
            ['\u01c9'] = "lj",
            ['\u01ca'] = "NJ",
            ['\u01cb'] = "Nj",
            ['\u01cc'] = "nj",
            ['\u01f1'] = "DZ",
            ['\u01f2'] = "Dz",
            ['\u01f3'] = "dz",
            ['\ufb00'] = "ff",
            ['\ufb01'] = "fi",
            ['\ufb02'] = "fl",
            ['\ufb03'] = "ffi",
            ['\ufb04'] = "ffl",
            ['\ufb05'] = "st",
            ['\ufb06'] = "st",
        };

        public static string Fix(string text) {
            // ftfy leaves HTML entities alone in text that looks like HTML.
            if (!text.Contains('<')) {
                text = WebUtility.HtmlDecode(text);
            }
            text = TerminalEscapes.Replace(text, "");
            text = FixLigaturesAndWidth(text);
            text = SingleQuotes.Replace(text, "'");
            text = DoubleQuotes.Replace(text, "\"");
            text = LineBreaks.Replace(text, "\n");
            text = ControlChars.Replace(text, "");
            return text.Normalize(NormalizationForm.FormC);
        }

        /// <summary>
        /// Replaces ligatures such as 'ﬁ' and full- or half-width characters such as 'Ａ' with their NFKC form.
        /// </summary>
        private static string FixLigaturesAndWidth(string text) {
            StringBuilder result = null;
            for (int i = 0; i < text.Length; i++) {
                char c = text[i];
                string replacement = null;
                if (Ligatures.TryGetValue(c, out string ligature)) {
                    replacement = ligature;
                } else if (c >= '\uff01' && c <= '\uff5e') {
                    // Full-width ASCII.
                    replacement = ((char)(c - 0xfee0)).ToString();
                } else if (c >= '\uff5f' && c <= '\uffef') {
                    string nfkc = c.ToString().Normalize(NormalizationForm.FormKC);
                    if (nfkc != c.ToString()) {
                        replacement = nfkc;
                    }
                } else if (c == '\u3000') {
                    replacement = " ";
                }
                if (replacement != null) {
                    result ??= new StringBuilder(text, 0, i, text.Length);
                    result.Append(replacement);
                } else {
                    result?.Append(c);
                }
            }
            return result?.ToString() ?? text;
        }
    }
}
