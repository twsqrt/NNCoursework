﻿using NeuralNetworks.ComputationGraph;
using NeuralNetworks.Transfer;

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
    Network Build();
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

        public Network Build()
        {
            var network = new Network(_input, _parameters.ToArray(), _currentRoot);

            _parameters.Clear();

            return network;
        }

        public ISpecifyActivationFunctionOrOutput ToLayerWithoutBias(int numberOfNeurons)
        {
            ParameterNode weights = ParameterNode.CreateRandom(numberOfNeurons * _currentRoot.Dimension);
            var layer = new LayerNode(weights, _currentRoot, 1);

            _currentRoot = layer;
            _parameters.Add(weights);

            return this;
        }

        private void AddBias(int numberOfNeurons)
        {
            ParameterNode bias = ParameterNode.CreateRandom(numberOfNeurons);
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
