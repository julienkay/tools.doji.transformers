using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Decides when generation ends, as transformers' stopping criteria do.
    /// </summary>
    public abstract class StoppingCriteria {

        /// <param name="inputIds">The prompt and the tokens generated so far.</param>
        public abstract bool IsDone(IReadOnlyList<int> inputIds);
    }

    /// <summary>
    /// Ends generation when the sequence, prompt included, has <see cref="MaxLength"/> tokens.
    /// </summary>
    public class MaxLengthCriteria : StoppingCriteria {

        public int MaxLength { get; }

        public MaxLengthCriteria(int maxLength) {
            MaxLength = maxLength;
        }

        public override bool IsDone(IReadOnlyList<int> inputIds) => inputIds.Count >= MaxLength;
    }

    /// <summary>
    /// Ends generation after <see cref="MaxTime"/> seconds.
    /// </summary>
    public class MaxTimeCriteria : StoppingCriteria {

        public float MaxTime { get; }
        private readonly Stopwatch _time = Stopwatch.StartNew();

        public MaxTimeCriteria(float maxTime) {
            MaxTime = maxTime;
        }

        public override bool IsDone(IReadOnlyList<int> inputIds) => _time.Elapsed.TotalSeconds > MaxTime;
    }

    /// <summary>
    /// Ends generation when the last token is an end-of-sequence token.
    /// </summary>
    public class EosTokenCriteria : StoppingCriteria {

        public int[] EosTokenId { get; }

        public EosTokenCriteria(int[] eosTokenId) {
            EosTokenId = eosTokenId;
        }

        public override bool IsDone(IReadOnlyList<int> inputIds) => EosTokenId.Contains(inputIds[^1]);
    }

    /// <summary>
    /// Ends generation when any of its criteria is met.
    /// </summary>
    public class StoppingCriteriaList : List<StoppingCriteria> {

        public bool IsDone(IReadOnlyList<int> inputIds) => this.Any(criteria => criteria.IsDone(inputIds));
    }
}
