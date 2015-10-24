using Encog.App.Analyst;
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
        private FileInfo OpenSetFile()
        {
            var dialog = new OpenFileDialog();
            dialog.Filter = "Comma separated values files (*.csv)|*.csv";

            if (dialog.ShowDialog() == true)
                return new FileInfo(dialog.FileName);
            return null;
        }

        private void Load_Train_Click(object sender, RoutedEventArgs e)
        {
            var fileInfo = OpenSetFile();
            if (fileInfo != null)
            {
                ViewModel.LoadTrainingSet(fileInfo);
            }
        }
        private void Load_Test_Click(object sender, RoutedEventArgs e)
        {
            var fileInfo = OpenSetFile();
            if (fileInfo != null)
            {
                ViewModel.LoadTestingSet(fileInfo);
            }
        }

        private void Train_Click(object sender, RoutedEventArgs e)
        {
            ViewModel.Train();
        }

        private void ClassificationRadio_Checked(object sender, RoutedEventArgs e)
        {
            ViewModel.ProblemType = AnalystGoal.Classification;
        }
        private void RegressionRadio_Checked(object sender, RoutedEventArgs e)
        {
            ViewModel.ProblemType = AnalystGoal.Regression;
        }

        private void SaveResults_Click(object sender, RoutedEventArgs e)
        {

        }

        private void Test_Click(object sender, RoutedEventArgs e)
        {
            ViewModel.Test();
        }
    }
}
