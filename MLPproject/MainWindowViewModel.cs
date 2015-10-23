using Encog;
using Encog.App.Analyst;
using Encog.App.Analyst.CSV.Normalize;
using Encog.App.Analyst.Wizard;
using Encog.Engine.Network.Activation;
using Encog.MathUtil;
using Encog.ML;
using Encog.ML.Data;

using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Persist;
using Encog.Util.Arrayutil;
using Encog.Util.CSV;
using Encog.Util.Logging;
using Encog.Util.Simple;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLPproject
{
    public enum ProblemType { Classifying, Regression };
    public class MainWindowViewModel : BindableBase
    {
        public MainWindowViewModel()
        {
            NormalizationTypes = new List<NormalizationAction> { NormalizationAction.Equilateral, NormalizationAction.OneOf };
            NormalizationType = NormalizationAction.Equilateral;
        }
        #region Properties
            #region Network
            public int NeuronsPerLayer { get { return neuronsPerLayer; } set { SetProperty(ref neuronsPerLayer, value); } }
            private int neuronsPerLayer = 3;

            public int NumberOfLayers  { get { return numberOfLayers; } set { SetProperty(ref numberOfLayers, value); } }
            private int numberOfLayers = 1;

            public double Bias { get { return bias; } set { SetProperty(ref bias, value); } }
            private double bias = 1;
            private bool HasBias { get { return(Bias != 0); } }

            public int NumberOfIterations { get { return numberOfIterations; } set { SetProperty(ref numberOfIterations, value); } }
            private int numberOfIterations = 100;

            public List<NormalizationAction> NormalizationTypes;
            public NormalizationAction NormalizationType { get { return normalizationType; } set { SetProperty(ref normalizationType, value); } }
            private NormalizationAction normalizationType = NormalizationAction.OneOf;

            public List<IActivationFunction> ActivationFunctions { get { return new List<IActivationFunction> { new ActivationTANH(), new ActivationRamp() }; } }
            public IActivationFunction Function { get { return function; } set { SetProperty(ref function, value); } }
            private IActivationFunction function;
        
            public double LearningRate { get { return learningRate; } set { SetProperty(ref learningRate, value); } }
            private double learningRate = 0.01;

            public double Momentum { get { return momentum; } set { SetProperty(ref momentum, value); } }
            private double momentum = 0.00;

            private BasicNetwork _network;
            #endregion
            #region Data Loading 

            public IMLDataSet TrainingData
            {
                get { return trainingData; }
                set
                {
                    SetProperty(ref trainingData, value);
                    OnPropertyChanged(nameof(IsTrainingDataLoaded));
                }
            }
            private IMLDataSet trainingData;
            public bool IsTrainingDataLoaded { get { return (TrainingData != null); } }

            public IMLDataSet TestingData
            {
                get { return testingData; }
                set
                {
                    SetProperty(ref testingData, value);
                    OnPropertyChanged(nameof(IsTestingDataLoaded));
                }
            }
            private IMLDataSet testingData;
            public bool IsTestingDataLoaded { get { return (TestingData != null); } }


            #endregion
            #region Visibility

            private bool isBusy = false;
            public bool IsBusy { get { return isBusy; } set { SetProperty(ref isBusy, value); } }
            public bool IsIdle { get { return !IsBusy; } }


            
            #endregion
            #region Error data
            public ObservableCollection<Tuple<int,double>> Progress { get { return progress; } set { SetProperty(ref progress, value); } }
            private ObservableCollection<Tuple<int,double>> progress = new ObservableCollection<Tuple<int, double>>();

            public double TrainingErrorValue { get { return trainingErrorValue; } set { SetProperty(ref trainingErrorValue, value); } }
            private double trainingErrorValue;
            public double TestingErrorValue { get { return testingErrorValue; } set { SetProperty(ref testingErrorValue, value); } }
            private double testingErrorValue;
        #endregion
        #endregion

        #region Classification
        public void LoadClassificationTrainingData(FileInfo fileInfo)
        {
            TrainingData = CSVHelper.LoadAndNormalizeData(fileInfo, AnalystGoal.Classification, this.NormalizationType, true);
        }
        public void LoadClassificationTestingData(FileInfo fileInfo)
        {
            TestingData = CSVHelper.LoadAndNormalizeData(fileInfo, AnalystGoal.Classification, this.NormalizationType, false);
        }
        public void TestClassification()
        {
            TestingErrorValue = _network.CalculateError(TestingData);
            var results = new List<double[]>();
            //FIX magic number
            var eq = new Equilateral(3, 1, -1);
            foreach (var singleResult in TestingData)
            {
                var input = new double[TestingData.InputSize+1];
                for (int i = 0; i < TestingData.InputSize; i++)
                    input[i] = singleResult.Input[i];
                input[TestingData.InputSize] = eq.Decode(singleResult.Ideal);
                results.Add(input);
            }
            CSVHelper.SaveToCSV(results, new FileInfo("temp/results.csv"));
        }

        #endregion
        #region Regresssion
        public void LoadRegressionTrainingData(FileInfo fileInfo)
        {
            TrainingData = CSVHelper.LoadAndNormalizeData(fileInfo, AnalystGoal.Regression, this.NormalizationType, true);
        }
        public void LoadRegressionTestingData(FileInfo fileInfo)
        {
            TestingData = CSVHelper.LoadAndNormalizeData(fileInfo, AnalystGoal.Regression, this.NormalizationType, false);
        }
        public void TestRegression()
        {
            var results = new List<double[]>();
            foreach (var singleResult in TestingData)
            {
                var input = new double[TestingData.InputSize + 2];
                for (int i = 0; i < TestingData.InputSize; i++)
                    input[i] = singleResult.Input[i];
                //FIX MagicNumber 0
                input[TestingData.InputSize] = singleResult.Ideal[0];
                input[TestingData.InputSize + 1] = _network.Compute(singleResult.Input)[0];
                results.Add(input);
            }
            TestingErrorValue = _network.CalculateError(TestingData);
            CSVHelper.SaveToCSV(results, new FileInfo("temp/regressionResults.csv"));
        }

        #endregion
        

        public void Train()
        {
            Progress.Clear();
            _network = ConstructNetwork(TrainingData.InputSize,TrainingData.IdealSize);

            //var trainer = new Backpropagation(_network, TrainingData, LearningRate, Momentum);
            var trainer = new ResilientPropagation(_network, TrainingData);
            double[] resultsArray = new double[TrainingData.Count];
            double[] errorArray = new double[NumberOfIterations];
            IsBusy = true;
            for (int iteration = 0; iteration < numberOfIterations; iteration++)
            {
                trainer.Iteration();
                Progress.Add(new Tuple<int,double>(iteration, trainer.Error));
            }
            IsBusy = false;
            for(int i = 0; i < TrainingData.Count; i++)
            {
               resultsArray[i] = _network.Classify(TrainingData[i].Input); 
            }
            TrainingErrorValue = _network.CalculateError(TrainingData);
        }
        private BasicNetwork ConstructNetwork(int inputNeurons, int outputNeurons)
        {
            //return EncogUtility.SimpleFeedForward(inputNeurons, inputNeurons, 0, outputNeurons, true);
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, HasBias, inputNeurons));
            for(int i = 0; i < NumberOfLayers; i++)
            {
                network.AddLayer(new BasicLayer(Function, HasBias, NeuronsPerLayer + (HasBias ? 1:0) ));
            }
            network.AddLayer(new BasicLayer(Function, false, outputNeurons));
            network.Structure.FinalizeStructure();
            network.Reset();
            return network;
        }

    }
}
