using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLPproject
{
    public enum ProblemType { Classifying, Regression };
    public class MainWindowViewModel : BindableBase
    {
        public int NeuronsPerLayer { get { return neuronsPerLayer; } set { SetProperty(ref neuronsPerLayer, value); } }
        private int neuronsPerLayer = 4;

        public int NumberOfLayers  { get { return numberOfLayers; } set { SetProperty(ref numberOfLayers, value); } }
        private int numberOfLayers = 1;

        public double Bias { get { return bias; } set { SetProperty(ref bias, value); } }
        private double bias = 0;

        public int NumberOfIterations { get { return numberOfIterations; } set { SetProperty(ref numberOfIterations, value); } }
        private int numberOfIterations = 100;

        public double LearningCoefficient { get { return learningCoefficient; } set { SetProperty(ref learningCoefficient, value); } }
        private double learningCoefficient = 1;

        public double MomentOfInertia { get { return momentOfInertia; } set { SetProperty(ref momentOfInertia, value); } }
        private double momentOfInertia = 1;

        public bool IsBusy { get { return isBusy; } set { SetProperty(ref isBusy, value); } }
        private bool isBusy = false;





    }
}
