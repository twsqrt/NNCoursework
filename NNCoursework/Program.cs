using NeuralNetworks.Network;
using NNCoursework;

const string PROJECT_DIRECTORY = @"C:\Users\twsqrt\source\repos\NNCoursework";
string networksFileDirectory = Path.Combine(PROJECT_DIRECTORY, "Networks");

// Console.Write("Number of epochs: ");
// int numberOfEpochs = Convert.ToInt32(Console.ReadLine());

// Console.Write("Learning rate: ");
// float learningRate = (float) Convert.ToDouble(Console.ReadLine());

// Console.Write("Weight decay: ");
// float weigthDecay = (float) Convert.ToDouble(Console.ReadLine());

// NeuralNetwork network = Train.TrainNetwork(numberOfEpochs, learningRate, weigthDecay);

// Console.WriteLine("Exporting network");
// using(var stream = File.OpenWrite(Path.Combine(networksFileDirectory, "test.bin")))
// using(var writer = new BinaryWriter(stream))
//     network.Export(writer);

NeuralNetwork network2 = null;
Console.WriteLine("Importing network");
using(var stream = File.OpenRead(Path.Combine(networksFileDirectory, "test.bin")))
using(var reader = new BinaryReader(stream))
    network2 = NeuralNetwork.Import(reader);

Console.WriteLine($"Test rate: {Test.NetworkRateOnTest(network2)}");
