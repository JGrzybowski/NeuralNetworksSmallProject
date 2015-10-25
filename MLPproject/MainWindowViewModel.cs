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
    public enum Stage { Start = 0, ProblemTypeSelected = 1, TrainingSetLoaded = 2, Trained = 3, TestingSetLoaded = 4 }
    public class MainWindowViewModel : BindableBase
    {
        public MainWindowViewModel()
        {
            NormalizationTypes = new List<NormalizationAction> { NormalizationAction.Equilateral, NormalizationAction.OneOf };
            NormalizationType = NormalizationAction.Equilateral;
        }
        #region Properties
        public AnalystGoal ProblemType
        {
            get { return problemType; }
            set { SetProperty(ref problemType, value); Stage = Stage.ProblemTypeSelected; }
        }
        private AnalystGoal problemType;
        public Stage Stage { get { return stage; } set { SetProperty(ref stage, value); } }
        private Stage stage = Stage.Start;

        #region Network Parameters
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

        public List<IActivationFunction> ActivationFunctions { get { return new List<IActivationFunction> {new ActivationTANH(), new ActivationLOG(), new ActivationLinear() }; } }
        public IActivationFunction Function { get { return function; } set { SetProperty(ref function, value); } }
        private IActivationFunction function;
        
        public double LearningRate { get { return learningRate; } set { SetProperty(ref learningRate, value); } }
        private double learningRate = 0.01;

        public double Momentum { get { return momentum; } set { SetProperty(ref momentum, value); } }
        private double momentum = 0.00;

        private BasicNetwork _network;
        #endregion
        #region Data Loading 

        public IMLDataSet TrainingSet
        {
            get { return trainingData; }
            set
            {
                SetProperty(ref trainingData, value);
                OnPropertyChanged(nameof(IsTrainingDataLoaded));
            }
        }
        private IMLDataSet trainingData;
        public bool IsTrainingDataLoaded { get { return (TrainingSet != null); } }

        public IMLDataSet TestingSet
        {
            get { return testingData; }
            set
            {
                SetProperty(ref testingData, value);
                OnPropertyChanged(nameof(IsTestingDataLoaded));
            }
        }
        private IMLDataSet testingData;
        public bool IsTestingDataLoaded { get { return (TestingSet != null); } }
        
        #endregion
        #region Visibility

        private bool isBusy = false;
        public bool IsBusy { get { return isBusy; } set { SetProperty(ref isBusy, value); } }
        public bool IsIdle { get { return !IsBusy; } }

        public string TestingSetFileName { get { return testingDataFileName; } set { SetProperty(ref testingDataFileName, value); } }
        private string testingDataFileName;
        public string TrainingSetFileName { get { return trainingDataFileName; } set { SetProperty(ref trainingDataFileName, value); } }
        private string trainingDataFileName;
        #endregion
        #region Error data
        public ObservableCollection<Tuple<int,double>> TrainingErrorData { get { return trainingErrorData; } set { SetProperty(ref trainingErrorData, value); } }
        private ObservableCollection<Tuple<int,double>> trainingErrorData = new ObservableCollection<Tuple<int, double>>();

        public ObservableCollection<Tuple<double, double>> TestingIdealData { get { return testingIdealData; } set { SetProperty(ref testingIdealData, value); } }
        private ObservableCollection<Tuple<double, double>> testingIdealData = new ObservableCollection<Tuple<double, double>>();

        public ObservableCollection<Tuple<double, double>> TestingResultsData { get { return testingResultsData; } set { SetProperty(ref testingResultsData, value); } }
        private ObservableCollection<Tuple<double, double>> testingResultsData = new ObservableCollection<Tuple<double, double>>();

        public List<ObservableCollection<Tuple<double, double>>> ClassPoints { get { return classPoints; } set { SetProperty(ref classPoints, value); } }
        private List<ObservableCollection<Tuple<double, double>>> classPoints = new List<ObservableCollection<Tuple<double, double>>>()
        {
             new ObservableCollection<Tuple<double, double>>(),
             new ObservableCollection<Tuple<double, double>>(),
             new ObservableCollection<Tuple<double, double>>(),
             new ObservableCollection<Tuple<double, double>>(),
             new ObservableCollection<Tuple<double, double>>(),
             new ObservableCollection<Tuple<double, double>>(),
             new ObservableCollection<Tuple<double, double>>(),
             new ObservableCollection<Tuple<double, double>>(),
        };

        private void ClearClassPoints()
        {
            foreach (var Class in ClassPoints)
            {
                Class.Clear();
            }
        }


        public double TrainingErrorValue { get { return trainingErrorValue; } set { SetProperty(ref trainingErrorValue, value); } }
        private double trainingErrorValue;
        public double TestingErrorValue { get { return testingErrorValue; } set { SetProperty(ref testingErrorValue, value); } }
        private double testingErrorValue;
        #endregion
        #endregion
        #region Classification
        public void TestClassification()
        {
            ClearClassPoints();
            TestingErrorValue = _network.CalculateError(TestingSet);

            var eq = new Equilateral(TrainingSet.IdealSize+1, 1, -1);
            foreach (var item in TrainingSet)
            {
                if (NormalizationType == NormalizationAction.OneOf)
                    ClassPoints[_network.Classify(item.Input)].Add(new Tuple<double, double>(item.Input[0], item.Input[1]));
                else
                {
                    var computedClass = new double[TrainingSet.IdealSize];
                    _network.Compute(item.Input).CopyTo(computedClass, 0, TrainingSet.IdealSize);
                    ClassPoints[eq.GetSmallestDistance(computedClass)].Add(new Tuple<double, double>(item.Input[0], item.Input[1]));
                }
            }
        }
        #endregion
        #region Regresssion
        public void TestRegression()
        {
            TestingIdealData.Clear();
            TestingResultsData.Clear();
            TestingErrorValue = _network.CalculateError(TestingSet);

            foreach (var item in TestingSet)
            {
                TestingIdealData.Add(new Tuple<double, double>(item.Input[0], item.Ideal[0]));
                TestingResultsData.Add(new Tuple<double,double>(item.Input[0], _network.Compute(item.Input)[0]));
            }

            TestingErrorValue = _network.CalculateError(TestingSet);
        }
        #endregion

        public void LoadTrainingSet(FileInfo fileInfo)
        {
            TrainingSet = CSVHelper.LoadAndNormalizeData(fileInfo, ProblemType, this.NormalizationType, true);
            TrainingSetFileName = fileInfo.Name;
            Stage = Stage.TrainingSetLoaded;
        }
        public void LoadTestingSet(FileInfo fileInfo)
        {
            TestingSet = CSVHelper.LoadAndNormalizeData(fileInfo, ProblemType, this.NormalizationType, false);
            TestingSetFileName = fileInfo.Name;
            Stage = Stage.TestingSetLoaded;
        }
        public void Train()
        {
            TrainingErrorData.Clear();
            TestingIdealData.Clear();
            TestingResultsData.Clear();
            _network = ConstructNetwork(TrainingSet.InputSize,TrainingSet.IdealSize);

            //var trainer = new Backpropagation(_network, TrainingSet, LearningRate, Momentum);
            var trainer = new ResilientPropagation(_network, TrainingSet);
            double[] resultsArray = new double[TrainingSet.Count];
            double[] errorArray = new double[NumberOfIterations];
            IsBusy = true;
            for (int iteration = 0; iteration < numberOfIterations; iteration++)
            {
                trainer.Iteration();
                TrainingErrorData.Add(new Tuple<int,double>(iteration, trainer.Error));
            }
            IsBusy = false;
            for(int i = 0; i < TrainingSet.Count; i++)
            {
               resultsArray[i] = _network.Classify(TrainingSet[i].Input); 
            }
            TrainingErrorValue = _network.CalculateError(TrainingSet);
            Stage = Stage.Trained;
        }
        public void Test()
        {
            if (ProblemType == AnalystGoal.Classification)
                TestClassification();
            else if(ProblemType == AnalystGoal.Regression)
                TestRegression();
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
