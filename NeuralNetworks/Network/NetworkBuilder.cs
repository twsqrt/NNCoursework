using NeuralNetworks.ComputationGraph;
using NeuralNetworks.Activation;

namespace NeuralNetworks.Network;

public interface ISpecifyInput
{
    ISpecifyLayer WithInput(int numberOfNeurons);
}

public interface ISpecifyLayer
{
    ISpecifyActivationFunctionOrOutput ToLayer(int numberOfNeurons);
    ISpecifyActivationFunctionOrOutput ToLayerWithoutBias(int numberOfNeurons);
}

public interface ISpecifyActivationFunction
{
    ISpecifyNextLayerOrOutput WithActivationFunction(ActivationType type);
}

public interface ISpecifyOutput
{
    IBuild ToOutput();
}

public interface ISpecifyNextLayerOrOutput 
    : ISpecifyLayer, ISpecifyOutput {}

public interface ISpecifyActivationFunctionOrOutput 
    : ISpecifyActivationFunction, ISpecifyOutput {}

public interface IBuild
{
    NeuralNetwork Build();
}

public class NetworkBuilder
{
    private class Iner
    : ISpecifyInput, ISpecifyNextLayerOrOutput, ISpecifyActivationFunctionOrOutput, IBuild
    {
        private readonly List<ParameterNode> _parameters;
        private ParameterNode _input;
        private Node _currentRoot;

        public Iner()
        {
            _parameters = new List<ParameterNode>();
            _input = null;
            _currentRoot = null;
        }

        public NeuralNetwork Build()
        {
            var network = new NeuralNetwork(_input, _parameters.ToArray(), _currentRoot);

            _parameters.Clear();

            return network;
        }

        public ISpecifyActivationFunctionOrOutput ToLayerWithoutBias(int numberOfNeurons)
        {
            bool isFistLayer = _parameters.Count == 0;

            ParameterNode weights = ParameterNode.CreateRandom(numberOfNeurons * _currentRoot.Dimension, -1.0f, 1.0f);
            var layer = new LayerNode(weights, _currentRoot, 1, ! isFistLayer);

            _currentRoot = layer;
            _parameters.Add(weights);

            return this;
        }

        private void AddBias(int numberOfNeurons)
        {
            ParameterNode bias = ParameterNode.CreateRandom(numberOfNeurons, -1.0f, 1.0f);
            var add = new AdditionNode(_currentRoot, bias, 1);

            _currentRoot = add;
            _parameters.Add(bias);
        }

        public ISpecifyActivationFunctionOrOutput ToLayer(int numberOfNeurons)
        {
            ToLayerWithoutBias(numberOfNeurons);
            AddBias(numberOfNeurons);

            return this;
        }

        public IBuild ToOutput() => this;

        public ISpecifyLayer WithInput(int numberOfNeurons)
        {
            _input = ParameterNode.CreateZero(numberOfNeurons);
            _currentRoot = _input;

            return this;
        }

        public ISpecifyNextLayerOrOutput WithActivationFunction(ActivationType type)
        {
            ActivationNode transfer = ActivationNode.Create(_currentRoot, type);
            _currentRoot = transfer;

            return this;
        }
    }

    private readonly Iner _iner;

    public NetworkBuilder()
    {
        _iner = new Iner();
    }

    public ISpecifyInput Create() => _iner;
}
