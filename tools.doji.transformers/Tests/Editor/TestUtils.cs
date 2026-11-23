using NUnit.Framework;
using System;
using System.Globalization;
using System.Linq;
using UnityEngine;

namespace Doji.AI.Transformers.Editor.Tests {

    public static class TestUtils {

        /// <summary>
        /// Reads a text file from Resources.
        /// </summary>
        public static string LoadText(string fileName) {
            TextAsset file = Resources.Load<TextAsset>(fileName);
            if (file == null) {
                throw new ArgumentException($"File '{fileName}' not found.");
            }
            return file.text;
        }

        /// <summary>
        /// Reads comma-separated numbers from a text file in Resources.
        /// </summary>
        public static float[] LoadFloats(string fileName) {
            return LoadText(fileName).Split(',').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
        }

        /// <inheritdoc cref="LoadFloats(string)"/>
        public static int[] LoadInts(string fileName) {
            return LoadText(fileName).Split(',').Select(v => int.Parse(v, CultureInfo.InvariantCulture)).ToArray();
        }

        /// <summary>
        /// Asserts two arrays are equal within a tolerance relative to max(1, |value|),
        /// and reports the largest difference when they are not.
        /// </summary>
        public static void AssertClose(float[] expected, float[] actual, float tolerance, string what) {
            Assert.That(actual.Length, Is.EqualTo(expected.Length), $"{what}: length");
            int worst = -1;
            float worstError = 0f;
            for (int i = 0; i < expected.Length; i++) {
                float error = Math.Abs(expected[i] - actual[i]) / Math.Max(1f, Math.Abs(expected[i]));
                if (!(error <= worstError)) {
                    worst = i;
                    worstError = error;
                }
            }
            Assert.That(worstError <= tolerance, $"{what}: largest error {worstError} at index {worst} " +
                $"(expected {(worst >= 0 ? expected[worst] : 0)}, got {(worst >= 0 ? actual[worst] : 0)}), tolerance {tolerance}");
        }
    }
}
