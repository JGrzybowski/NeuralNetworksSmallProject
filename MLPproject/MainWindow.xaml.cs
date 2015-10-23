using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MLPproject
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        MainWindowViewModel ViewModel => this.DataContext as MainWindowViewModel;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void LoadTrainingSet_Click(object sender, RoutedEventArgs e)
        {
            var fileInfo = OpenSetFile();
            if (fileInfo != null)
                ViewModel.LoadClassificationTrainingData(fileInfo);        
        }

        private FileInfo OpenSetFile()
        {
            var dialog = new OpenFileDialog();
            dialog.Filter = "Comma separated values files (*.csv)|*.csv";

            if (dialog.ShowDialog() == true)
                return new FileInfo(dialog.FileName);
            return null;
        }

        private void StartTraining_Click(object sender, RoutedEventArgs e)
        {
            ViewModel.Train();
        }

        private void TestClassification_Click(object sender, RoutedEventArgs e)
        {
            var fileInfo = OpenSetFile();
            if (fileInfo != null)
            {
                ViewModel.LoadClassificationTestingData(fileInfo);
                ViewModel.TestClassification();
            }

        }

        private void Load_Train_Regression_Click(object sender, RoutedEventArgs e)
        {
            var fileInfo = OpenSetFile();
            if (fileInfo != null)
            {
                ViewModel.LoadRegressionTrainingData(fileInfo);
            }
        }

        private void TestRegression_Click(object sender, RoutedEventArgs e)
        {
            var fileInfo = OpenSetFile();
            if (fileInfo != null)
            {
                ViewModel.LoadRegressionTestingData(fileInfo);
                ViewModel.TestRegression();
            }
        }
    }
}
