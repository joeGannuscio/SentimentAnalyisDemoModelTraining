using Microsoft.ML.Data;

namespace SentimentAnalyisDemoModelTraining.Models
{
    /// <summary>
    /// This class is the prediction results
    /// </summary>
    public class SentimentPrediction
    {
        /// <summary>
        /// This is the predicited negativity flag from the model
        /// </summary>
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }
}
