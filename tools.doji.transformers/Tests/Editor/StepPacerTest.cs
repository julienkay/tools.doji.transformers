using NUnit.Framework;

namespace Doji.AI.Transformers.Editor.Tests {

    public class StepPacerTest {

        private const float Frame = 1 / 60f;

        [Test]
        public void DoublesUntilAFrameIsSlow_ThenAddsOne() {
            var pacer = new StepPacer(new GenerationPacing());
            pacer.EndFrame(0.5f, -1, limited: true); // the prompt
            Assert.That(pacer.StepsPerFrame, Is.EqualTo(1));
            pacer.EndFrame(Frame, 3, limited: true);
            pacer.EndFrame(Frame, 3, limited: true);
            pacer.EndFrame(Frame, 3, limited: true);
            Assert.That(pacer.StepsPerFrame, Is.EqualTo(8));
            pacer.EndFrame(2 * Frame, 3, limited: true);
            Assert.That(pacer.StepsPerFrame, Is.EqualTo(4));
            pacer.EndFrame(Frame, 3, limited: true);
            Assert.That(pacer.StepsPerFrame, Is.EqualTo(5));
        }

        [Test]
        public void HalvesWhenATokenComesBackLate() {
            var pacer = new StepPacer(new GenerationPacing());
            pacer.EndFrame(Frame, -1, limited: true);
            pacer.EndFrame(Frame, 3, limited: true);
            pacer.EndFrame(Frame, 3, limited: true);
            Assert.That(pacer.StepsPerFrame, Is.EqualTo(4));
            pacer.EndFrame(Frame, StepPacer.ReadbackFrames + 2, limited: true);
            Assert.That(pacer.StepsPerFrame, Is.EqualTo(2));
        }

        [Test]
        public void DoesNotGrow_WhenNotLimitedByTheSteps() {
            var pacer = new StepPacer(new GenerationPacing());
            pacer.EndFrame(Frame, -1, limited: true);
            pacer.EndFrame(Frame, 3, limited: false);
            Assert.That(pacer.StepsPerFrame, Is.EqualTo(1));
        }

        [Test]
        public void StaysWithinMaxStepsPerFrame() {
            var pacer = new StepPacer(new GenerationPacing { MaxStepsPerFrame = 3 });
            for (int i = 0; i < 10; i++) {
                pacer.EndFrame(Frame, 3, limited: true);
            }
            Assert.That(pacer.StepsPerFrame, Is.EqualTo(3));
            Assert.That(pacer.InFlight, Is.EqualTo(3 * (StepPacer.ReadbackFrames + 1)));
        }

        [Test]
        public void GenerationPacing_RejectsLimitsThatAllowNoWork() {
            Assert.That(() => new GenerationPacing { MaxStepsPerFrame = 0 }, Throws.TypeOf<System.ArgumentOutOfRangeException>());
            Assert.That(() => new GenerationPacing { MaxMillisecondsPerFrame = 0 }, Throws.TypeOf<System.ArgumentOutOfRangeException>());
        }
    }
}
