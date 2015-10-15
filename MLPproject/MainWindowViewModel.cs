using Encog;
using Encog.App.Analyst;
using Encog.App.Analyst.CSV.Normalize;
using Encog.App.Analyst.Wizard;
using Encog.Engine.Network.Activation;
using Encog.ML;
using Encog.ML.Data;

using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Persist;
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
        public FileInfo DataFile { get { return dataFile; } set { SetProperty(ref dataFile, value); } }
        private FileInfo dataFile;

        public ObservableCollection<Tuple<int,double>> Progress { get { return progress; } set { SetProperty(ref progress, value); } }
        private ObservableCollection<Tuple<int,double>> progress = new ObservableCollection<Tuple<int, double>>();
        
        public int NeuronsPerLayer { get { return neuronsPerLayer; } set { SetProperty(ref neuronsPerLayer, value); } }
        private int neuronsPerLayer = 2;

        public int NumberOfLayers  { get { return numberOfLayers; } set { SetProperty(ref numberOfLayers, value); } }
        private int numberOfLayers = 1;

        public double Bias { get { return bias; } set { SetProperty(ref bias, value); } }
        private double bias = 1;
        private bool HasBias { get { return(Bias != 0); } }

        public int NumberOfIterations { get { return numberOfIterations; } set { SetProperty(ref numberOfIterations, value); } }
        private int numberOfIterations = 1000;

        public IActivationFunction Function { get { return new ActivationBiPolar(); } }

        public double LearningRate { get { return learningRate; } set { SetProperty(ref learningRate, value); } }
        private double learningRate = 0.75;

        public double Momentum { get { return momentum; } set { SetProperty(ref momentum, value); } }
        private double momentum = 0.5;

        public bool IsBusy { get { return isBusy; } set { SetProperty(ref isBusy, value); } }
        private bool isBusy = false;

        public double ErrorValue { get { return errorValue; } set { SetProperty(ref errorValue, value); } }
        private double errorValue;

        public void Train()
        {
            Progress.Clear();
            var analyst = new EncogAnalyst();
            var wizard = new AnalystWizard(analyst);
            wizard.Wizard(DataFile, true, AnalystFileFormat.DecpntComma);
            var fields = analyst.Script.Normalize.NormalizedFields;
            fields[fields.Count - 1].Action = Encog.Util.Arrayutil.NormalizationAction.Equilateral;
            
            var norm = new AnalystNormalizeCSV();
            norm.Analyze(DataFile, true, CSVFormat.DecimalPoint, analyst);
            norm.Normalize(new FileInfo("temp.csv"));

            var inputNeurons = fields.Count-1;
            var outputNeurons = fields.Last().Classes.Count-1;
            var trainingSet = TrainingSetUtil.LoadCSVTOMemory(CSVFormat.DecimalPoint,"temp.csv",true, inputNeurons, outputNeurons);
            
            var network = ConstructNetwork(inputNeurons,outputNeurons);
            //var s = new FileStream("machine.egb", FileMode.Create);
            //Encog.Persist.EncogDirectoryPersistence.SaveObject(s, network);
            //EncogDirectoryPersistence.SaveObject(new FileInfo("machine.eg"), network);
            var trainer = new Backpropagation(network, trainingSet, LearningRate, Momentum);

            double[] resultsArray = new double[trainingSet.Count];
            double[] errorArray = new double[NumberOfIterations];
            for (int iteration = 0; iteration < numberOfIterations; iteration++)
            {
                trainer.Iteration();
                errorValue = trainer.Error;
                Progress.Add(new Tuple<int,double>(iteration, trainer.Error));
            }
            for(int i =0; i< trainingSet.Count; i++)
            {
               resultsArray[i] = network.Classify(trainingSet[i].Input);
            }
        }
         
        private IMLDataSet LoadCSV(FileInfo fileInfo)
        {
            int numberOfColumns = 0;
            if (fileInfo.Exists)
            {
                using (StreamReader sr = new StreamReader(fileInfo.FullName))
                    numberOfColumns = sr.ReadLine().Split(',').Count();
            
                return TrainingSetUtil.LoadCSVTOMemory(CSVFormat.DecimalPoint, "temp.csv", true, numberOfColumns-2, 1);
            }
            throw new FileNotFoundException("File not found", fileInfo.FullName);
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
