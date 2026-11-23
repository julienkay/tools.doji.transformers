using System;
using System.Threading.Tasks;

namespace Doji.AI.Transformers {

    /// <summary>
    /// The sync API runs the async code with <c>sync: true</c>, where every await completes at once.
    /// </summary>
    internal static class SyncTask {

        /// <summary>
        /// Gets the result of a task that ran synchronously.
        /// </summary>
        public static T Run<T>(Task<T> task) {
            if (!task.IsCompleted) {
                throw new InvalidOperationException("A synchronous call had to wait. Use the async method instead.");
            }
            return task.GetAwaiter().GetResult();
        }
    }
}
