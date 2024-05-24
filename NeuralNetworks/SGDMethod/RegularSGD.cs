using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class RegularSGD : ISGDMethod
{
    private readonly float _learningRate;

    public RegularSGD(float learningRate)
    {
        _learningRate = learningRate;
    }

    public float CalculateLearningRate(ParameterNode[] parameter, Vector<float>[] gradient)
        => _learningRate;
}
