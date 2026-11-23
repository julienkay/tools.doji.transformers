using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Applies a list of <see cref="LogitsProcessor"/>s in order.
    /// </summary>
    public class LogitsProcessorList : List<LogitsProcessor> {

        public void Process(IReadOnlyList<int> inputIds, float[] scores) {
            foreach (LogitsProcessor processor in this) {
                processor.Process(inputIds, scores);
            }
        }
    }
}
