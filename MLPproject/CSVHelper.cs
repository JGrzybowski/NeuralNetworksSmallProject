using Encog.App.Analyst;
using Encog.App.Analyst.Wizard;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Microsoft.VisualBasic.FileIO;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog.Util.Arrayutil;
using Encog.App.Analyst.CSV.Normalize;
using Encog.Util.CSV;

namespace MLPproject
{
    public static class CSVHelper
    {
        public static IMLDataSet LoadAndNormalizeData(FileInfo fileInfo, AnalystGoal problemType, NormalizationAction normalizationType, bool randomize = true)
        {
            var analyst = new EncogAnalyst();
            var wizard = new AnalystWizard(analyst);
            wizard.Goal = problemType;
            wizard.Wizard(fileInfo, true, AnalystFileFormat.DecpntComma);
            var fields = analyst.Script.Normalize.NormalizedFields;

            if (problemType == AnalystGoal.Classification)
                fields[fields.Count - 1].Action = normalizationType;

            var norm = new AnalystNormalizeCSV();
            norm.Analyze(fileInfo, true, CSVFormat.DecimalPoint, analyst);

            var normalizedDataFileInfo = new FileInfo("temp/temp.csv");
            norm.Normalize(normalizedDataFileInfo);

            var inputNeurons = fields.Count - 1;
            int outputNeurons;
            if (problemType == AnalystGoal.Classification)
                outputNeurons = fields.Last().Classes.Count - (normalizationType == NormalizationAction.Equilateral ? 1 : 0);
            else
                outputNeurons = fields.Count - inputNeurons;
            var result = CSVHelper.LoadCSVToDataSet(normalizedDataFileInfo, inputNeurons, outputNeurons, randomize);
            normalizedDataFileInfo.Delete();
            return result;
        }

        public static IMLDataSet LoadCSVToDataSet(FileInfo fileInfo, int inputCount, int outputCount, bool randomize = true, bool headers = true)
        {
            BasicMLDataSet result = new BasicMLDataSet();
            CultureInfo CSVformat = new CultureInfo("en");

            using (TextFieldParser parser = new TextFieldParser(fileInfo.FullName))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");
                if (headers)
                    parser.ReadFields();
                while (!parser.EndOfData)
                {
                    //Processing row
                    string[] fields = parser.ReadFields();
                    var input = new BasicMLData(inputCount);
                    for (int i = 0; i < inputCount; i++)
                        input[i] = double.Parse(fields[i], CSVformat);
                    var ideal = new BasicMLData(outputCount);
                    for (int i = 0; i < outputCount; i++)
                        ideal[i] = double.Parse(fields[i + inputCount], CSVformat);
                    result.Add(input, ideal);
                }
            }
            var rand = new Random(DateTime.Now.Millisecond);

            return (randomize ? new BasicMLDataSet(result.OrderBy(r => rand.Next()).ToList()) : new BasicMLDataSet(result));
        }


        public static void SaveToCSV(List<double[]> array, FileInfo fileInfo)
        {
            using (var sw = fileInfo.CreateText())
            {
                foreach (var row in array)
                {
                    sw.WriteLine(String.Join(";",row));
                }
                sw.Close();
            }
        }


    }
}
