using UnityEngine;
using UnityEngine.UIElements;

namespace Doji.AI.Transformers.Samples.TextGeneration {

    /// <summary>
    /// A small icon of the chat UI, drawn with <see cref="Painter2D"/> in the text color of the element, so the
    /// sample needs no image assets. Size it in USS.
    /// </summary>
    [UxmlElement]
    public partial class ChatIcon : VisualElement {

        public enum Shape { Send, Stop, NewChat, Sparkle, ChevronDown }

        private Shape _shape;

        [UxmlAttribute]
        public Shape IconShape {
            get => _shape;
            set {
                _shape = value;
                MarkDirtyRepaint();
            }
        }

        public ChatIcon() : this(Shape.Sparkle) { }

        public ChatIcon(Shape shape) {
            _shape = shape;
            AddToClassList("chat-icon");
            pickingMode = PickingMode.Ignore;
            generateVisualContent += Draw;
            RegisterCallback<CustomStyleResolvedEvent>(_ => MarkDirtyRepaint());
        }

        private void Draw(MeshGenerationContext context) {
            Rect r = contentRect;
            if (r.width <= 0 || r.height <= 0) {
                return;
            }
            // The shapes are on a 24x24 grid, centered in the element.
            float scale = Mathf.Min(r.width, r.height) / 24f;
            Vector2 origin = r.center - new Vector2(12f, 12f) * scale;
            Vector2 P(float x, float y) => origin + new Vector2(x, y) * scale;

            var p = context.painter2D;
            Color color = resolvedStyle.color;
            p.strokeColor = color;
            p.fillColor = color;
            p.lineWidth = 2.2f * scale;
            p.lineCap = LineCap.Round;
            p.lineJoin = LineJoin.Round;

            switch (_shape) {
                case Shape.Send:
                    p.BeginPath();
                    p.MoveTo(P(12, 19));
                    p.LineTo(P(12, 5.5f));
                    p.MoveTo(P(6, 11));
                    p.LineTo(P(12, 5));
                    p.LineTo(P(18, 11));
                    p.Stroke();
                    break;
                case Shape.Stop:
                    p.BeginPath();
                    p.MoveTo(P(9, 7));
                    p.ArcTo(P(17, 7), P(17, 17), 2f * scale);
                    p.ArcTo(P(17, 17), P(7, 17), 2f * scale);
                    p.ArcTo(P(7, 17), P(7, 7), 2f * scale);
                    p.ArcTo(P(7, 7), P(17, 7), 2f * scale);
                    p.ClosePath();
                    p.Fill();
                    break;
                case Shape.NewChat:
                    p.BeginPath();
                    p.MoveTo(P(12, 5));
                    p.LineTo(P(12, 19));
                    p.MoveTo(P(5, 12));
                    p.LineTo(P(19, 12));
                    p.Stroke();
                    break;
                case Shape.Sparkle:
                    p.BeginPath();
                    p.MoveTo(P(12, 2));
                    p.QuadraticCurveTo(P(13, 11), P(22, 12));
                    p.QuadraticCurveTo(P(13, 13), P(12, 22));
                    p.QuadraticCurveTo(P(11, 13), P(2, 12));
                    p.QuadraticCurveTo(P(11, 11), P(12, 2));
                    p.ClosePath();
                    p.Fill();
                    break;
                case Shape.ChevronDown:
                    p.lineWidth = 2f * scale;
                    p.BeginPath();
                    p.MoveTo(P(7, 10));
                    p.LineTo(P(12, 15));
                    p.LineTo(P(17, 10));
                    p.Stroke();
                    break;
            }
        }
    }
}
