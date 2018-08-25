using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using WikiDetoxBinaryClassification.Models.Sentiment;


namespace WikiDetoxBinaryClassification
{
    class Program
    {

        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static async Task Main(string[] args)
        {
            var model = await Train();
            Evaluate(model);
            Predict(model);
        }

        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            var pipeline = new LearningPipeline();

            // Carga o ingiere los datos.
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());

            // Preprocesa y caracteriza los datos.
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier()
            {
                NumLeaves = 5,
                NumTrees = 5,
                MinDocumentsInLeafs = 2
            });

            // Entrena el modelo.
            PredictionModel<SentimentData, SentimentPrediction> model =
                pipeline.Train<SentimentData, SentimentPrediction>();

            // Predice sentimientos en función de datos de prueba. 
            await model.WriteAsync(_modelpath);
            return model;
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            // Carga el conjunto de datos de prueba.
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();

            // Crea el evaluador binario.
            var evaluator = new BinaryClassificationEvaluator();

            // Evalúa el modelo y crea métricas.
            evaluator.Evaluate(model, testData);

            // Muestra las métricas.
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        public static void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            // Crea datos de prueba.
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Please refrain from adding nonsense to Wikipedia."
                },
                new SentimentData
                {
                    SentimentText = "He is the best, and the article should say that."
                }
            };

            // Predice sentimientos en función de datos de prueba.
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);

            var sentimentsAndPredictions = sentiments
            .Zip(predictions,
                (sentiment, prediction) => (sentiment, prediction));
            // Combina datos de prueba y predicciones para la generación de informes.
            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");
            foreach (var item in predictions)
            {
                Console.WriteLine($"Sentiment: {(item.Sentiment ? "Positive" : "Negative")}");
            }

            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }

            Console.WriteLine();

            // Muestra los resultados de la predicción.
        }
    }


}
