using System.Reflection.Metadata;
using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Network;

public class Network
{
    private static readonly Random random = new Random();
    private readonly ParameterNode[] _parameters;
    private readonly ParameterNode _input;
    private readonly Node _output;
    private readonly Node _index;
    private readonly ParameterNode _indexParameter;
    private readonly Vector<float>[] _cachedGradient;

    public int InputDimension => _input.Dimension;
    public int OutputDimension => _output.Dimension;

    public Network(ParameterNode input, ParameterNode[] parameters, Node output)
    {
        _parameters = parameters;
        _input = input; 
        _output = output;

        _indexParameter = ParameterNode.CreateZero(output.Dimension);
        _index = new MetricNode(_output, _indexParameter, 1);

        _cachedGradient = new Vector<float>[_parameters.Length];
        for(int i = 0; i < _cachedGradient.Length; i++)
        {
            int dimension = _parameters[i].Dimension;
            _cachedGradient[i] = Vector<float>.CreateZeroVector(dimension);
        }
    }

    private void CreateBatch(Vector<float>[] data, Vector<float>[] markup, int batchSize)
    {
        foreach(Vector<float> parameterGradient in _cachedGradient)
            parameterGradient.SetZero();

        for(int j = 0; j < batchSize; j++)
        {
            int index = random.Next(0, data.Length);

            _input.Value = data[index];
            _indexParameter.Value = markup[index];

            _index.CalculateValue();
            _index.Backpropagate();
            
            for(int k = 0; k < _parameters.Length; k++)
            {
                Vector<float> parameterGradient = _parameters[k].CurrentJacobian.AsVector();
                parameterGradient.Scale(1.0f / batchSize);

                _cachedGradient[k].Add(parameterGradient);
            }
        }
    }

    public void Fit(Vector<float>[] data, Vector<float>[] markup, int batchSize, int iterationCount, ISGDMethod sgdMethod, Action<float> progressCallback)
    {
        int percentInteger = 0;

        for(int i = 0; i < iterationCount; i++)
        {
            float percent = 100 * i / iterationCount;
            if(percent + 1.0f > percentInteger)
            {
                progressCallback(percent);
                percentInteger++;
            }

            CreateBatch(data, markup, batchSize);

            float learningRate = sgdMethod.CalculateLearningRate(_parameters, _cachedGradient);

            for(int j = 0; j < _parameters.Length; j++)
            {
                _cachedGradient[j].Scale(-1.0f * learningRate);
                _parameters[j].Value.Add(_cachedGradient[j]);
            }
        }
    }

    public void Fit(Vector<float>[] data, Vector<float>[] markup, int batchSize, int iterationCount, ISGDMethod sgdMethod, TextWriter log)
        => Fit(data, markup, batchSize, iterationCount, sgdMethod, percent => {
            if(percent % 10 == 0)
                log.WriteLine($"Progress: {percent}%");
        });

    public Vector<float> Execute(Vector<float> input)
    {
        _input.Value = input;
        return _output.CalculateValue();
    }
}
