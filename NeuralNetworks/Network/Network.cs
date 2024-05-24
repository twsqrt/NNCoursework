using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Network;

public class Network
{
    private readonly ParameterNode[] _parameters;
    private readonly ParameterNode _input;
    private readonly Node _output;
    private readonly Node _index;
    private readonly ParameterNode _indexParameter;

    public Network(ParameterNode input, ParameterNode[] parameters, Node output)
    {
        _parameters = parameters;
        _input = input; 
        _output = output;

        _indexParameter = ParameterNode.CreateZero(output.Dimension);
        _index = new MetricNode(_output, _indexParameter);
    }

    public void Fit(TrainData[] data, int iterationCount, float learningRate)
    {
        var random = new Random();
        for(int i = 0; i < iterationCount; i++)
        {
            int index = random.Next(0, data.Length);
            TrainData row = data[index];

            _input.Value = row.Data;
            _indexParameter.Value = row.Markup;

            _index.CalculateValue();
            _index.Backpropagate();

            foreach(ParameterNode parameter in _parameters)
            {
                Vector<float> gradient = parameter.CurrentJacobian.AsVector();
                gradient.Scale(-1.0f * learningRate);

                parameter.Value.Add(gradient);
            }
        }
    }

    public Vector<float> Execute(Vector<float> input)
    {
        _input.Value = input;
        return _output.CalculateValue();
    }
}
