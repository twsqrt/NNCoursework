using NeuralNetworks.Network;
using NNCoursework;

Console.Write("Number of epochs: ");
int numberOfEpochs = Convert.ToInt32(Console.ReadLine());

Console.Write("Learning rate: ");
float learningRate = (float) Convert.ToDouble(Console.ReadLine());

Console.Write("Weight decay: ");
float weigthDecay = (float) Convert.ToDouble(Console.ReadLine());

NeuralNetwork network = Train.TrainNetwork(numberOfEpochs, learningRate, weigthDecay);