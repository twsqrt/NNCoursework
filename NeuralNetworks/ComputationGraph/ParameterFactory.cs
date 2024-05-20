using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public static class ParameterFactory
{
    private static readonly Random _random = new Random();

    public static Parameter CreateFromData(float[] data)
        => new Parameter(new Vector<float>(data));

    public static Parameter CreateZero(int dimension)
        => new Parameter(Vector<float>.ZeroVector(dimension));
    
    public static Parameter CreateRandom(int dimension)
    {
        var data = new float[dimension];
        for(int i = 0; i < dimension; i++)
            data[i] = (float)_random.NextDouble();

        return CreateFromData(data);
    }
}
