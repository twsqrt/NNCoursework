using LinearAlgebra;
using NeuralNetworks.ComputationGraph;


namespace NeuralNetworks.Network;

public class NeuralNetwork
{
    private readonly VectorInputNode _input;
    private readonly IDataNode[] _parameters;
    private readonly INode[] _nodes;
    private readonly Node<Vector> _output;
    private readonly LossNode _loss;
    private readonly VectorInputNode _lossMarkup;

    public NeuralNetwork(VectorInputNode input, IDataNode[] parameters, Node<Vector> output)
    {
        _input = input;
        _parameters = parameters;
        _output = output;

        _lossMarkup = new VectorInputNode(output.Shape.Dimension);
        _loss = new LossNode(output, _lossMarkup);

        var nodesList = new List<INode>{_loss};
        for(int i = 0; i < nodesList.Count(); i++)
        {
            INode node = nodesList[i];

            foreach(INode parameter in node.Parameters)
            {
                if(parameter is not IDataNode)
                    nodesList.Add(parameter);
            }
        }

        nodesList.Reverse();
        _nodes = nodesList.ToArray();
    }

    private void UpdateValues()
    {
        for(int i = 0; i < _nodes.Length; i++)
            _nodes[i].CalculateValue();
    }

    private void Backpropagate()
    {
        for(int i = _nodes.Length - 1; i >= 0; i--)
            _nodes[i].CalculateGradient();
    }

    public void Fit(Vector[] data, Vector[] markup, float learningRate, float weigthDecay, Action<int> progressCallback)
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

            _input.Value = data[i];
            _lossMarkup.Value = markup[i];

            UpdateValues();
            Backpropagate();

            if (float.IsNaN(_loss.Value[0]))
                throw new ArgumentException();

            foreach(IDataNode parameter in _parameters)
            {
                float[] gradient = parameter.GradientData;
                for(int j = 0; j < parameter.Data.Length; j++)
                {
                    float old = parameter.Data[j];
                    parameter.Data[j] -= learningRate * gradient[j] + learningRate * weigthDecay * old;
                }
            }
        }
    }

    public void Fit(Vector[] data, Vector[] markup, float learningRate, float weigthDecay, TextWriter log)
        => Fit(data, markup, learningRate, weigthDecay, percent => {
            if(percent % 5 == 0)
                log.WriteLine($"Train progress: {percent}%");
        });
    
    public void Fit(Vector[] data, Vector[] markup, float learningRate, float weigthDecay, int numberOfEpochs, TextWriter log)
    {
        for(int i = 0; i < numberOfEpochs; i++)
        {
            log.WriteLine($"Epoch number: {i + 1}");
            Fit(data, markup, learningRate, weigthDecay, log);
        }
    }

    public Vector Execute(Vector input)
    {
        _input.Value = input;
        UpdateValues();
        return _output.Value;
    }
}
