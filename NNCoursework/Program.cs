using LinearAlgebra;
using NeuralNetworks.Network;
using NNCoursework;

MnistData.LoadTrainData();
MnistData.LoadTestData();

while (true)
{
    Console.Write("Train/test/exit? ");
    string response = Console.ReadLine();

    if (response.ToLower() == "train")
    {
        Console.Write("Number of epochs: ");
        int numberOfEpochs = int.Parse(Console.ReadLine());

        Console.Write("Learning rate: ");
        float learningRate = float.Parse(Console.ReadLine());

        Console.Write("Create new? (yes/no) ");
        bool shouldCreateNewNetwork = Console.ReadLine() == "yes";

        int startNumberOfEpochs = 0;
        string networkName = string.Empty;

        NeuralNetwork network;
        if (shouldCreateNewNetwork)
            network = NetworkLoader.CreateNewNetwork();
        else
        {
            Console.Write("Snapshot file name: ");
            string shapshotFileName = Console.ReadLine();
            network = NetworkLoader.LoadShapshot(shapshotFileName);

            string[] parts = shapshotFileName.Split("_epoch=");
            networkName = parts[0];
            startNumberOfEpochs = int.Parse(parts[1]);
        }

        Console.Write("Create snapshots every epoch? (yes/no) ");
        bool shouldCreateShaphots = Console.ReadLine() == "yes";

        bool shouldExportResult = shouldCreateShaphots;
        if (shouldCreateShaphots == false)
        {
            Console.Write("Export results? (yes/no) ");
            shouldExportResult = Console.ReadLine() == "yes";
        }

        if (shouldExportResult && networkName == string.Empty)
        {
            Console.Write("Network name: ");
            networkName = Console.ReadLine();
        }

        Console.Write("Exit after: (yes/no) ");
        bool shouldExitAfter = Console.ReadLine() == "yes";


        Vector[] data, markup;
        MnistData.GetTrainData(out data, out markup);

        if (shouldCreateShaphots)
        {
            Train.TrainNetwork(network, networkName, numberOfEpochs, learningRate, epoch =>
            {
                Console.WriteLine();
                Console.WriteLine($"Train result: {Test.TestNetwork(network, 5000, data, markup)}");
                Console.WriteLine($"Test result: {Test.TestNetwork(network)}");

                if (shouldCreateShaphots)
                    NetworkLoader.CreateShapshot(network, networkName, epoch + startNumberOfEpochs);
            });
        }
        else
        {
            Train.TrainNetwork(network, networkName, numberOfEpochs, learningRate, _ => { });
        }

        if (shouldExportResult && !shouldCreateShaphots)
            NetworkLoader.CreateShapshot(network, networkName, numberOfEpochs);

        if (shouldExitAfter)
            break;
    }
    else if (response.ToLower() == "test")
    {
        Console.Write("Snapshot file name: ");
        string name = Console.ReadLine();
        NeuralNetwork network = NetworkLoader.LoadShapshot(name);

        Vector[] data, markup;
        MnistData.GetTrainData(out data, out markup);

        Console.WriteLine($"Train result: {Test.TestNetwork(network, 5000, data, markup)}");
        Console.WriteLine($"Test result: {Test.TestNetwork(network)}");
    }
    else if (response.ToLower() == "exit")
        break;
}