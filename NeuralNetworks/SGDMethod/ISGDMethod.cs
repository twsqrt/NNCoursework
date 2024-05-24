using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public interface ISGDMethod
{
    float CalculateLearningRate(ParameterNode[] parameter, Vector<float>[] gradient);
}
