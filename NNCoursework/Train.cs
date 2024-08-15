using LinearAlgebra;
using NeuralNetworks.Network;
using NNCoursework;

public static class Train
{
    public static void TrainNetwork(NeuralNetwork network, 
        string networkName,
        int numberOfEpochs, 
        float learningRate,
        Action<int> epochCallback)
    {
        Vector[] data, markup;
        MnistData.GetTrainData(out data, out markup);

        for(int i = 0; i < numberOfEpochs; i++)
        {
            Console.WriteLine($"Epoch: {i + 1} / {numberOfEpochs}");

            network.Fit(data, markup, learningRate, 0.0f, (progress) => {
                Console.SetCursorPosition(0, Console.CursorTop);
                Console.Write($"Train progress: {progress}%");
            });

            epochCallback(i + 1);
        }
    }
}
