using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;

namespace RealNeuralNetworkMNIST
{
    class Program2
    {
        const int IMAGE_SIZE = 28;              //each image 28*28 pixels
        const int IMAGE_VECTOR_SIZE = IMAGE_SIZE * IMAGE_SIZE;//data length image 28*28 pixels = 784 px
        const int SAMPLE_COUNT = 10;            //analyse 10 images - numbers 0..9
        const int TRAIN_ROWS_COUNT = 1900;      //first rows to train;
        const int TEST_ROWS_COUNT = 100;       //other rows to test
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test_200_rows.csv";//43% 100+100
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test_2000_rows.csv";//53% 1000+1000
        const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test_2000_rows.csv";//56% 1900+100
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//50% 9900+100
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//56% 9000+1000
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//57% 5000+5000
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//49% 1000+9000
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//41% 100+9900
        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_train.csv";//55% 5000+5000 black/white

        //const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";//16% 59900+100
        //надо тупо сделать фулл коннектед слои каждый с каждым: 784*10*10 
        //а не раздельные десяток пучков по 784 нейронов в каждом

        private static double[]  layerInputNodes = new double[IMAGE_VECTOR_SIZE];
        private static double[,] layerAssociationsNodes = new double[SAMPLE_COUNT, IMAGE_VECTOR_SIZE];
        private static double[,] layerAssociationsWeights = new double[SAMPLE_COUNT, IMAGE_VECTOR_SIZE];
        private static double[,] layerAssociationsWeightDeltas = new double[SAMPLE_COUNT, IMAGE_VECTOR_SIZE];
        private static double[]  layerResultNodes = new double[SAMPLE_COUNT];
        private static double[,] layerResultWeights = new double[SAMPLE_COUNT, IMAGE_VECTOR_SIZE];
        private static double[,] layerResultWeightPartialDeltas = new double[SAMPLE_COUNT, IMAGE_VECTOR_SIZE];
        private static double[,] layerResultWeightDeltas = new double[SAMPLE_COUNT, IMAGE_VECTOR_SIZE];

        private static double learningRate = 0.0001;//если 0.5 то слишком сильно меняет вес для 1ой цифры
        private static long correctResults = 0;
        private static double error;

        static void Main2(string[] args)
        {
            initWeights();
            train();
            test();

            Console.WriteLine("Правильно распознано {0}% вариантов",
                100 * correctResults / TEST_ROWS_COUNT);
        }

        //оно меняет так веса, что запоминает первую цифру в обучающей выборке - 7
        //если skip сделать 2, то начнет менять веса под цифру 2
        //а на следующих итерациях, веса уже недостаточно меняются

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
            var rows = File.ReadAllLines(FILE_PATH).Skip(1 + TRAIN_ROWS_COUNT).Take(TEST_ROWS_COUNT).ToList();

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
            for (int i = 0; i < SAMPLE_COUNT; i++)
            {
                for (int j = 0; j < IMAGE_VECTOR_SIZE; j++)
                {
                    layerAssociationsWeights[i, j] = /*0.001;*/rand.NextDouble()*0.001;
                    layerResultWeights[i, j] = /*0.001;*/rand.NextDouble() *0.001;
                }
            }
        }

        private static void calculateStatistics(int correctNumber)
        {
            var max = layerResultNodes.Max();
            var proposalNumber = 0;
            for (int i = 0; i < SAMPLE_COUNT; i++)
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
            for (int i = 0; i < SAMPLE_COUNT; i++)
            {
                var target = (i == correctNumber) ? 1 : 0/*0.5*/;
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
                layerInputNodes[i - 1] = double.Parse(values[i]) / byte.MaxValue;// делить на 1000?
                //layerInputNodes[i - 1] = funActivation(double.Parse(values[i]) / byte.MaxValue);                
            }
        }

        private static void calculateAssociationsLayer()
        {
            for (int i = 0; i < SAMPLE_COUNT; i++)
            {
                for (int j = 0; j < IMAGE_VECTOR_SIZE; j++)
                {
                    //if (layerInputNodes[j] == 0) continue;//add
                    layerAssociationsNodes[i, j] =
                        funActivation(layerInputNodes[j] 
                        * layerAssociationsWeights[i, j]);
                }
            }
        }

        private static void calculateResultLayer()
        {
            for (int i = 0; i < SAMPLE_COUNT; i++)
            {
                layerResultNodes[i] = 0;
                for (int j = 0; j < IMAGE_VECTOR_SIZE; j++) 
                {
                    layerResultNodes[i] += (layerAssociationsNodes[i, j] * layerResultWeights[i, j]);                                    
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
            for (int i = 0; i < SAMPLE_COUNT; i++)
            {
                var target = (i == correctNumber) ? 1 : 0/*0.5*/;
                var actual = layerResultNodes[i];

                for (int j = 0; j < IMAGE_VECTOR_SIZE; j++)
                {
                    layerResultWeightPartialDeltas[i, j] =
                        (target - actual) * actual * (1 - actual);
                    layerResultWeightDeltas[i, j] = 
                        layerResultWeightPartialDeltas[i, j]
                        * layerAssociationsNodes[i, j];
                }
            }
        }

        private static void calculateLayerAssociationsWeightDeltas()
        {
            for (int i = 0; i < SAMPLE_COUNT; i++)
            {
                for (int j = 0; j < IMAGE_VECTOR_SIZE; j++)
                {
                    var sumOutgoingLinks = 
                        layerResultWeightPartialDeltas[i, j] 
                        * layerResultWeights[i, j]; //just one link in this case

                    layerAssociationsWeightDeltas[i, j] =
                        sumOutgoingLinks
                        * layerAssociationsNodes[i, j]
                        * (1 - layerAssociationsNodes[i, j])
                        * layerInputNodes[j];
                }
            }
        }

        private static void updateLayerResultWeightDeltas()
        {
            for (int i = 0; i < SAMPLE_COUNT; i++)
            {
                for (int j = 0; j < IMAGE_VECTOR_SIZE; j++)
                {
                    layerResultWeights[i, j] += (learningRate * layerResultWeightDeltas[i, j]);
                }
            }
        }

        private static void updateLayerAssociationsWeightDeltas()
        {
            for (int i = 0; i < SAMPLE_COUNT; i++)
            {
                for (int j = 0; j < IMAGE_VECTOR_SIZE; j++)
                {
                    layerAssociationsWeights[i, j] += (learningRate * layerAssociationsWeightDeltas[i, j]);
                }
            }
        }
    }
}
