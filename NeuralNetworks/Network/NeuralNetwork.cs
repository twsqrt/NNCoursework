using LinearAlgebra;
using NeuralNetworks.ComputationGraph;
using NeuralNetworks.File;
using NeuralNetworks.SDGMethod;


namespace NeuralNetworks.Network;

public class NeuralNetwork
{
    private readonly ParameterNode[] _parameters;
    private readonly ParameterNode _input;
    private readonly Node _output;
    private readonly Node _index;
    private readonly ParameterNode _indexParameter;
    private readonly Vector<float>[] _cachedGradient;

    public ParameterNode Input => _input;
    public ParameterNode[] Parameters => _parameters;
    public Node Output => _output;

    public NeuralNetwork(ParameterNode input, ParameterNode[] parameters, Node output)
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

    public void Fit(Vector<float>[] data, Vector<float>[] markup, ISGDMethod sgdMethod, Action<int> progressCallback)
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
            _indexParameter.Value = markup[i];

            _index.CalculateValue();
            _index.Backpropagate();

            float learningRate = sgdMethod.CalculateLearningRate(_parameters, _cachedGradient);

            foreach(ParameterNode parameter in _parameters)
            {
                Vector<float> gradient = parameter.CurrentJacobian.AsVector();
                parameter.Value.Add(gradient.Scale(-1.0f * learningRate));
            }
        }
    }

    public void Fit(Vector<float>[] data, Vector<float>[] markup, ISGDMethod sgdMethod, TextWriter log)
        => Fit(data, markup, sgdMethod, percent => {
            if(percent % 5 == 0)
                log.WriteLine($"Train progress: {percent}%");
        });
    
    public void Fit(Vector<float>[] data, Vector<float>[] markup, ISGDMethod sgdMethod, int numberOfEpochs, TextWriter log)
    {
        for(int i = 0; i < numberOfEpochs; i++)
        {
            log.WriteLine($"Epoch number: {i + 1}");
            Fit(data, markup, sgdMethod, log);
        }
    }

    public Vector<float> Execute(Vector<float> input)
    {
        _input.Value = input;
        return _output.CalculateValue();
    }

    public static NeuralNetwork Import(BinaryReader reader)
        => NetworkFileManager.Read(reader);

    public void Export(BinaryWriter writer)
        => NetworkFileManager.Write(this, writer);
}
