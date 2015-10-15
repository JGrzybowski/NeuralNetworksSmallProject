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
            var dialog = new OpenFileDialog();
            dialog.Filter = "Comma separated values files (*.csv)|*.csv";

            if (dialog.ShowDialog() == true)
                ViewModel.DataFile = new FileInfo(dialog.FileName);        
        }

        private void StartTraining_Click(object sender, RoutedEventArgs e)
        {
            ViewModel.Train();
        }
    }
}
