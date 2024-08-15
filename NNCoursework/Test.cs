using LinearAlgebra;
using NeuralNetworks.Network;

namespace NNCoursework;

public static class Test
{
    public static float TestNetwork(NeuralNetwork network, int dataSize, Vector[] data, Vector[] markup)
    {
        int correctCount = 0;

        for(int i = 0; i < dataSize; i++)
        {
            Vector output = network.Execute(data[i]);

            float maxElement = 0.0f;
            int outputNumber = 0;
            for(int j = 0; j < output.Dimension; j++)
            {
                if(maxElement < output[j])
                {
                    maxElement = output[j];
                    outputNumber = j;
                }
            }

            if(markup[i][outputNumber] == 1.0f)
                correctCount++;
        }

        float rate = (float) correctCount / dataSize;
        return rate;
    }

    public static float TestNetwork(NeuralNetwork network)
    {
        Vector[] data, markup;
        MnistData.GetTestData(out data, out markup);
        return TestNetwork(network, data.Length, data, markup);
    }
}
