using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;

namespace RealNeuralNetworkMNIST
{
    class Program
    {
        //Скачать файлы для работы нейросети можно тут: https://www.kaggle.com/oddrationale/mnist-in-csv
        const int INPUT_LAYER_SIZE = 784;       //each image 28*28 pixels = 784 px
        const int ASSOCIATIONS_LAYER_SIZE = 20;
        const int RESULT_LAYER_SIZE = 10;       //analyse 10 images - numbers 0..9

        const int INPUT_LAYER_LINKS_SIZE = INPUT_LAYER_SIZE * ASSOCIATIONS_LAYER_SIZE;//784 * 20 = 15680
        const int ASSOCIATIONS_LAYER_LINKS_SIZE = ASSOCIATIONS_LAYER_SIZE * RESULT_LAYER_SIZE;//20 * 10 = 200

        const int TRAIN_ROWS_COUNT = 5000;      //first rows to train;
        const int TEST_ROWS_COUNT = 5000;       //other rows to test

        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test_2000_rows.csv";//36% 1900+100
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//89% 9900+100
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//89% 10000+10000
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//89% 9000+1000
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test_200_rows.csv";//9% 100+100
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test_2000_rows.csv";//15% 1000+1000
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//80% 5000+5000
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//18% 1000+9000
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//10% 100+9900

        const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_train.csv";//94% 59900+100
        //const string TEST_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//91% 60.000+10.000
        const string TEST_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//97% 60.000+100

        private static double[] layerInputNodes = new double[INPUT_LAYER_SIZE];
        private static double[] layerAssociationsNodes = new double[ASSOCIATIONS_LAYER_SIZE];
        private static double[] layerAssociationsWeights = new double[INPUT_LAYER_LINKS_SIZE];
        private static double[] layerAssociationsWeightDeltas = new double[INPUT_LAYER_LINKS_SIZE];
        private static double[] layerResultNodes = new double[RESULT_LAYER_SIZE];
        private static double[] layerResultWeights = new double[ASSOCIATIONS_LAYER_LINKS_SIZE];
        private static double[] layerResultWeightPartialDeltas = new double[ASSOCIATIONS_LAYER_LINKS_SIZE];
        private static double[] layerResultWeightDeltas = new double[ASSOCIATIONS_LAYER_LINKS_SIZE];

        private static double learningRate = 0.5;
        private static long correctResults = 0;
        private static double error;

        static void Main(string[] args)
        {
            initWeights();
            train();
            test();

            Console.WriteLine("Правильно распознано {0}% вариантов",
                100 * correctResults / TEST_ROWS_COUNT);
        }

        private static void train()
        {
            Console.WriteLine("Начало тренировки нейросети");
            var index = 1;
            var rows = File.ReadAllLines(FILE_PATH).Skip(1).Take(TRAIN_ROWS_COUNT).ToList();

            foreach (var row in rows)
            {
                Console.WriteLine("Итерация {0} из {1}", index++, TRAIN_ROWS_COUNT);
                var values = row.Split(',');
                var correctNumber = byte.Parse(values[0]);

                assignInputNodesLayer(values);
                calculateAssociationsLayer();
                calculateResultLayer();
                backPropagation(correctNumber);
                printTotalError(correctNumber);
            }
        }

        private static void test()
        {
            Console.WriteLine("Начало тестирования нейросети");
            var index = 1;
            //var rows = File.ReadAllLines(FILE_PATH).Skip(1 + TRAIN_ROWS_COUNT).Take(TEST_ROWS_COUNT).ToList();
            var rows = File.ReadAllLines(TEST_PATH).Skip(1).Take(TEST_ROWS_COUNT).ToList();            

            foreach (var row in rows)
            {
                Console.WriteLine("Итерация {0} из {1}", index++, TEST_ROWS_COUNT);
                var values = row.Split(',');
                var correctNumber = byte.Parse(values[0]);

                assignInputNodesLayer(values);
                calculateAssociationsLayer();
                calculateResultLayer();
                calculateStatistics(correctNumber);
            }
        }

        private static void initWeights()
        {
            var rand = new Random();
            for (int i = 0; i < layerAssociationsWeights.Length; i++)
            {
                layerAssociationsWeights[i] = rand.NextDouble() * 0.001;
            }
            for (int i = 0; i < layerResultWeights.Length; i++)
            {
                layerResultWeights[i] = rand.NextDouble() * 0.001;
            }
        }

        private static void calculateStatistics(int correctNumber)
        {
            var max = layerResultNodes.Max();
            var proposalNumber = 0;
            for (int i = 0; i < layerResultNodes.Length; i++)
			{
                if (layerResultNodes[i] == max) {
                    proposalNumber = i;
                    break;
                }			 
			}
            Console.WriteLine("Число {0} определено как {1} {2}", correctNumber, proposalNumber,
                proposalNumber == correctNumber ? "УСПЕХ" : "НЕУДАЧА");
            if (proposalNumber == correctNumber) correctResults++;
        }

        private static void printTotalError(int correctNumber)
        {
            error = 0;
            for (int i = 0; i < layerResultNodes.Length; i++)
            {
                var target = (i == correctNumber) ? 1 : 0;
                error += (0.5 * Math.Pow(target - layerResultNodes[i], 2));
            }
            Console.WriteLine("Ошибка нейросети: {0}", error);
        }

        private static double funActivation(double value)
        {
            return (1 / (1 + Math.Pow(Math.E, -value)));
        }

        private static void assignInputNodesLayer(string[] values)
        {
            for (int i = 1; i < values.Length; i++)
            {
                layerInputNodes[i - 1] = double.Parse(values[i]) / byte.MaxValue;
            }
        }

        private static void calculateAssociationsLayer()
        {
            for (int i = 0; i < layerAssociationsNodes.Length; i++)
            {
                layerAssociationsNodes[i] = 0;
                for (int j = 0; j < layerInputNodes.Length; j++)
                {
                    layerAssociationsNodes[i] += 
                        (layerInputNodes[j]
                        * layerAssociationsWeights[(i * layerInputNodes.Length) + j]);
                }
                layerAssociationsNodes[i] = funActivation(layerAssociationsNodes[i]);
            }
        }

        private static void calculateResultLayer()
        {
            for (int i = 0; i < layerResultNodes.Length; i++)
            {
                layerResultNodes[i] = 0;
                for (int j = 0; j < layerAssociationsNodes.Length; j++) 
                {
                    layerResultNodes[i] += 
                        (layerAssociationsNodes[j]
                        * layerResultWeights[(i * layerAssociationsNodes.Length) + j]);                                    
                }
                layerResultNodes[i] = funActivation(layerResultNodes[i]);
            }
        }

        private static void backPropagation(int correctNumber)
        {
            calculateLayerResultWeightDeltas(correctNumber);
            calculateLayerAssociationsWeightDeltas();
            updateLayerResultWeightDeltas();
            updateLayerAssociationsWeightDeltas();
        }

        private static void calculateLayerResultWeightDeltas(int correctNumber)
        {
            for (int i = 0; i < layerResultNodes.Length; i++)
            {
                var target = (i == correctNumber) ? 1 : 0;
                var actual = layerResultNodes[i];

                for (int j = 0; j < layerAssociationsNodes.Length; j++)
                {
                    layerResultWeightPartialDeltas[(i * layerAssociationsNodes.Length) + j] =
                        (target - actual) * actual * (1 - actual);
                    layerResultWeightDeltas[(i * layerAssociationsNodes.Length) + j] =
                        layerResultWeightPartialDeltas[(i * layerAssociationsNodes.Length) + j]
                        * layerAssociationsNodes[j];
                }
            }
        }

        private static double getSumOutgoingLinks(int index)
        {
            double result = 0;
            for (int i = 0; i < layerResultNodes.Length; i++)
            {
                result +=
                    (layerResultWeightPartialDeltas[(layerAssociationsNodes.Length * i) + index]
                    * layerResultWeights[(layerAssociationsNodes.Length * i) + index]);
            }
            return result;
        }

        private static void calculateLayerAssociationsWeightDeltas()
        {
            for (int i = 0; i < layerAssociationsNodes.Length; i++)
            {
                var sumOutgoingLinks = getSumOutgoingLinks(i);
                for (int j = 0; j < layerInputNodes.Length; j++)
                {
                    layerAssociationsWeightDeltas[(i * layerInputNodes.Length) + j] =
                        sumOutgoingLinks
                        * layerAssociationsNodes[i]
                        * (1 - layerAssociationsNodes[i])
                        * layerInputNodes[j];
                }
            }
        }

        private static void updateLayerResultWeightDeltas()
        {
            for (int i = 0; i < layerResultWeights.Length; i++)
            {
                layerResultWeights[i] += (learningRate * layerResultWeightDeltas[i]);
            }
        }

        private static void updateLayerAssociationsWeightDeltas()
        {
            for (int i = 0; i < layerAssociationsWeights.Length; i++)
            {
                layerAssociationsWeights[i] += (learningRate * layerAssociationsWeightDeltas[i]);
            }
        }
    }
}
