using System;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Sets how many decode steps <see cref="PreTrainedModel.GenerateAsync(System.Collections.Generic.IReadOnlyList{int}, GenerationConfig, BaseStreamer, Random, System.Threading.CancellationToken)"/>
    /// schedules per frame, when the tokens are picked on the GPU. A small model can run several steps in one frame;
    /// a large one runs one step per frame.
    /// </summary>
    /// <remarks>
    /// It doubles the steps per frame until a frame is slow, then adds a step per frame while the frames stay short
    /// and the tokens come back in time. It halves the steps when a frame takes much longer than the shortest frame
    /// so far, or when a token comes back late. A token that comes back late means the GPU has more work queued than
    /// it does in a frame. <see cref="GenerationPacing"/> limits the steps per frame.
    /// </remarks>
    internal sealed class StepPacer {

        private readonly int _maxStepsPerFrame;

        /// <summary>
        /// The most main-thread time per frame for scheduling steps. The first step of a frame is always scheduled.
        /// </summary>
        public double MaxMillisecondsPerFrame { get; }

        public StepPacer(GenerationPacing pacing) {
            _maxStepsPerFrame = pacing.MaxStepsPerFrame;
            MaxMillisecondsPerFrame = pacing.MaxMillisecondsPerFrame;
        }

        /// <summary>The frames from the request of a readback to its data, when the GPU keeps up.</summary>
        public const int ReadbackFrames = 3;

        /// <summary>A frame is slow when it takes this many times the shortest frame so far.</summary>
        public const float SlowFrameFactor = 1.5f;

        /// <summary>The steps to schedule in the next frame.</summary>
        public int StepsPerFrame { get; private set; } = 1;

        /// <summary>
        /// The most steps whose token is not back yet: the steps of the frames that a readback takes, and one frame more.
        /// </summary>
        public int InFlight => Math.Max(4, StepsPerFrame * (ReadbackFrames + 1));

        private float _shortestFrame = float.PositiveInfinity;
        private int _frames;
        private bool _growing = true;

        /// <summary>
        /// Updates the steps per frame after a frame.
        /// </summary>
        /// <param name="deltaTime">The length of the frame, in seconds.</param>
        /// <param name="latency">The frames that the last token read in the frame took to come back, or -1.</param>
        /// <param name="limited">True when the frame scheduled <see cref="StepsPerFrame"/> steps, so more could help.</param>
        public void EndFrame(float deltaTime, int latency, bool limited) {
            // The first frame runs the prompt.
            if (_frames++ == 0) {
                return;
            }
            _shortestFrame = Math.Min(_shortestFrame, deltaTime);
            bool slow = deltaTime > SlowFrameFactor * _shortestFrame;
            bool late = latency > ReadbackFrames + 1;
            if (slow || late) {
                StepsPerFrame = Math.Max(1, StepsPerFrame / 2);
                _growing = false;
            } else if (limited) {
                // Like the slow start of TCP: double the steps until the first slow frame, then add one at a time.
                StepsPerFrame = Math.Min(_maxStepsPerFrame, _growing ? 2 * StepsPerFrame : StepsPerFrame + 1);
            }
        }
    }
}
