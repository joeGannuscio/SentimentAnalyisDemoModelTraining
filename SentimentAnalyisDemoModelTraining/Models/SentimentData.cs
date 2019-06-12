using Microsoft.ML.Data;

namespace SentimentAnalyisDemoModelTraining.Models
{
    /// <summary>
    /// This class is used to map the input data to parameters used to train the model
    /// </summary>
    public class SentimentData
    {
        /// <summary>
        /// This is the flag from the labeled data that indicated if a comment was marked as negative or toxic
        /// </summary>
        [LoadColumn(2)]
        public bool IsNegative { get; set; }

        /// <summary>
        /// This is the text that was evaluated
        /// </summary>
        [LoadColumn(4)]
        public string Text { get; set; }
    }
}
