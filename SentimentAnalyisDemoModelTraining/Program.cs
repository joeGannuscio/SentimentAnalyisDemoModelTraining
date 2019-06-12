using Microsoft.ML;
using SentimentAnalyisDemoModelTraining.Models;

namespace SentimentAnalyisDemoModelTraining
{
    class Program
    {
        //specify the loaction of the input data file
        private const string InputDataFile = "";

        static void Main(string[] args)
        {
            //create a new MLContext which is used for all the model creation and training
            var mlContext = new MLContext();

            #region Data Cleaning and Prep

            //creates an IDataView from the input file using the SentimentData.
            //the data is lazy loaded, so no data will actually load until it is used 
            var inputData = mlContext.Data.LoadFromTextFile<SentimentData>(InputDataFile, hasHeader: true);

            //because sentiment analysis uses strings as an input, we have to 
            //convet the text into numbers that that algorithms can use
            //this is built into the framework in the FeaturizeText method
            var featurizedTextEstimator = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text));

            //split the dataset into training and validation sets
            //in this case we want 20% of the data to be used for model validation
            var trainTestDataSplit = mlContext.Data.TrainTestSplit(inputData, 0.2);

            var trainingData = trainTestDataSplit.TrainSet;
            var validationData = trainTestDataSplit.TestSet;

            //select the algorithm and create a trainer for the model
            var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression("IsNegative", "Features");

            #endregion

            #region Model Training

            //add the model to the training pipeline
            var modelTrainer = featurizedTextEstimator.Append(trainer);

            //run the model training process
            var trainedModel = modelTrainer.Fit(trainingData);

            #endregion

            #region Model Validation

            //use the trained model to make predictions based on the validation dataset
            var validation = trainedModel.Transform(validationData);

            //calculate the metrics on the trained model using the validation dataset
            var modelMetrics = mlContext.BinaryClassification.Evaluate(validation, "IsNegative", "Score");

            #endregion

            //save the model for future use
            mlContext.Model.Save(trainedModel, trainingData.Schema, "SentimentAnalysisDemoModel.zip");
        }
    }
}
