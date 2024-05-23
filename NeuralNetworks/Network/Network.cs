using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Network;

public class Network
{
    private readonly Parameter[] _parameters;
    private readonly Parameter _input;
    private readonly Node _root;
    private readonly Node _indexRoot;
    private readonly Parameter _indexParameter;

    public Network(Parameter input, Parameter[] parameters, Node root)
    {
        _parameters = parameters;
        _input = input; 
        _root = root;

        _indexParameter = ParameterFactory.CreateZero(_root.Dimension);
        _indexRoot = new SquareMetric(_indexParameter, _root);
    }

    public void Fit(TrainData[] data, int iterationCount, float learningRate)
    {
        var random = new Random();
        for(int i = 0; i < iterationCount; i++)
        {
            int index = random.Next(0, data.Length);
            TrainData row = data[index];

            _input.SetValue(row.Data);
            _indexParameter.SetValue(row.Markup);

            _indexRoot.UpdateValue();
            _indexRoot.Backpropagate();

            foreach(Parameter parameter in _parameters)
            {
                Vector<float> gradient = parameter.CurrentJacobian.ToVectorCached();
                gradient.MultiplyByScalar(learningRate);
                Vector<float> newValue = Vector<float>.Difference(parameter.CurrentValue, gradient);
                parameter.SetValue(newValue);
            }
        }
    }

    public IReadOnlyVector<float> Execute(IReadOnlyVector<float> input)
    {
        _input.SetValue(input);
        return _root.UpdateValue();
    }
}
