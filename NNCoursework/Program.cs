using NNCoursework;

Console.Write("train/test? ");

string reply = Console.ReadLine();
if(reply == "train")
{
    Console.Write("Number of epochs: ");
    int numberOfEpochs = Convert.ToInt32(Console.ReadLine());

    Console.Write("Learning rate: ");
    float learningRate = (float) Convert.ToDouble(Console.ReadLine());

    Train.TrainNetwork(numberOfEpochs, learningRate);
}

if(reply == "test")
{
    Console.Write("Network file name: ");
    string networkFileName = Console.ReadLine();

    Test.TestNetwork(networkFileName);
}