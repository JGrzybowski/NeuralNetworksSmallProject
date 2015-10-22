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

namespace MLPproject
{
    public static class CSVHelper
    {
        public static IMLDataSet LoadCSVToDataSet(FileInfo fileInfo, int inputCount, int outputCount, bool headers = true)
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
                        ideal[i] = double.Parse(fields[i+inputCount], CSVformat);
                    result.Add(input, ideal);
                }
            }
            var rand = new Random(DateTime.Now.Millisecond);
            return new BasicMLDataSet(result.OrderBy(r => rand.Next()).ToList());
        }



    }
}
