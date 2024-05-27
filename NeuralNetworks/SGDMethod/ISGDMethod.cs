using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.SDGMethod;

public interface ISGDMethod
{
    float CalculateLearningRate(ParameterNode[] parameter, Vector<float>[] gradient);
}
