using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class ParameterNode : Node
{
    private static readonly Random _random = new Random();

    private Vector _gradient;

    public Vector Value { get; set; }
    public Vector Gradient => _gradient;

    public ParameterNode(Vector value) : base(value.Dimension)
    {
        Value = value;
        _gradient = null;
    }

    public static ParameterNode CreateZero(int dimension)
        => new ParameterNode(Vector.CreateZero(dimension));
    
    public static ParameterNode CreateFromArray(float[] data)
        => new ParameterNode(new Vector(data));
    
    public static ParameterNode CreateRandom(int dimension, float cordMinValue = 0.0f, float cordMaxValue = 1.0f)
    {
        var data = new float[dimension];
        for(int i = 0; i < data.Length; i++)
        {
            float element = (float) _random.NextDouble() * (cordMaxValue - cordMinValue) + cordMinValue;
            data[i] = element;
        }

        return CreateFromArray(data);
    }

    public override void BackpropagateNext(Vector gradient)
    {
        _gradient = gradient;
    }

    public override Vector CalculateValue() => Value;

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);
}