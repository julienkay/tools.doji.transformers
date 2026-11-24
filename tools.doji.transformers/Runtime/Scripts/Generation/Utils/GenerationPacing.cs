using System;

namespace Doji.AI.Transformers {

    /// <summary>
    /// How much work <see cref="PreTrainedModel.GenerateAsync(System.Collections.Generic.IReadOnlyList{int}, GenerationConfig, BaseStreamer, Random, System.Threading.CancellationToken)"/>
    /// does per frame on the GPU backend. Lower the limits to leave more of each frame to the rest of the app.
    /// </summary>
    /// <remarks>
    /// This has no counterpart in transformers: it changes when the tokens are generated, not which tokens.
    /// Within the limits, the steps per frame adapt: they grow while the frames stay short and the tokens come back
    /// in time, and drop when a frame gets much longer than the shortest one or the GPU falls behind.
    /// </remarks>
    public class GenerationPacing {

        private int _maxStepsPerFrame = 32;
        private float _maxMillisecondsPerFrame = 4f;

        /// <summary>
        /// The most decode steps (tokens) per frame. 1 generates at most one token per frame.
        /// </summary>
        public int MaxStepsPerFrame {
            get => _maxStepsPerFrame;
            set => _maxStepsPerFrame = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(value), "Must be at least 1.");
        }

        /// <summary>
        /// The most main-thread time per frame, in milliseconds, for scheduling decode steps. A frame always
        /// schedules one step, even when that step takes longer.
        /// </summary>
        public float MaxMillisecondsPerFrame {
            get => _maxMillisecondsPerFrame;
            set => _maxMillisecondsPerFrame = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), "Must be more than 0.");
        }
    }
}
