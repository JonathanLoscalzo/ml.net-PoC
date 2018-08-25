using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace IrisClassification.Models
{
   public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }
}