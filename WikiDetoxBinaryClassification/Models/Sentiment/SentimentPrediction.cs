using Microsoft.ML.Runtime.Api;

namespace WikiDetoxBinaryClassification.Models.Sentiment
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
    }
}