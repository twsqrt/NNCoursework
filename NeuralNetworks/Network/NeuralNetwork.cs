using LinearAlgebra;
using NeuralNetworks.ComputationGraph;


namespace NeuralNetworks.Network;

public class NeuralNetwork
{
    private readonly DataNode _input;
    private readonly DataNode[] _parameters;
    private readonly Node<Vector> _output;
    private readonly LossNode _loss;
    private readonly DataNode _lossMarkup;

    public NeuralNetwork(DataNode input, DataNode[] parameters, Node<Vector> output)
    {
        _input = input;
        _parameters = parameters;
        _output = output;

        _lossMarkup = new DataNode(_output.Shape.Dimension);
        _loss = new LossNode(output, _lossMarkup);
    }

    public void Fit(Vector[] data, Vector[] markup, float learningRate, Action<int> progressCallback)
    {
        int percentInteger = 0;

        for(int i = 0; i < data.Length; i++)
        {
            float percent = 100.0f * i / data.Length;
            if(percent + 1.0f > percentInteger)
            {
                progressCallback(percentInteger);
                percentInteger++;
            }

            _input.Data = data[i];
            _lossMarkup.Data = markup[i];

            float loss = _loss.CalculateValue();
            if(float.IsNaN(loss))
                throw new ArgumentException();

            _loss.Backpropagate();

            foreach(DataNode parameter in _parameters)
            {
                Vector gradient = parameter.Gradient;
                gradient.Scale(-1.0f * learningRate);
                parameter.Data.Add(gradient);
            }
        }
    }

    public void Fit(Vector[] data, Vector[] markup, float learningRate, TextWriter log)
        => Fit(data, markup, learningRate, percent => {
            if(percent % 5 == 0)
                log.WriteLine($"Train progress: {percent}%");
        });
    
    public void Fit(Vector[] data, Vector[] markup, float learningRate, int numberOfEpochs, TextWriter log)
    {
        for(int i = 0; i < numberOfEpochs; i++)
        {
            log.WriteLine($"Epoch number: {i + 1}");
            Fit(data, markup, learningRate, log);
        }
    }

    public Vector Execute(Vector input)
    {
        _input.Data = input;
        return _output.CalculateValue();
    }
}
