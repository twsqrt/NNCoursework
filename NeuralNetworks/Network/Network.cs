using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Network;

public class Network
{
    private readonly Parameter[] _parameters;
    private readonly Parameter _input;
    private readonly Node _root;

    public Network(Parameter input, Parameter[] parameters, Node root)
    {
        _parameters = parameters;
        _input = input; 
        _root = root;
    }

    public void Fit(IReadOnlyVector<float>[] x, IReadOnlyVector<float>[] y)
    {
        throw new NotImplementedException();
    }

    public IReadOnlyVector<float> Execute(IReadOnlyVector<float> input)
    {
        _input.SetValue(input);
        return _root.UpdateValue();
    }
}
