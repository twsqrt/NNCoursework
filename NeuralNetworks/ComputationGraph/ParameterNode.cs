using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class ParameterNode : Node
{
    private static readonly Random _random = new Random();

    private Matrix _currentJacobian;

    public Vector Value { get; set; }
    public Matrix CurrentJacobian => _currentJacobian;

    public ParameterNode(Vector value) : base(value.Dimension)
    {
        Value = value;

        _currentJacobian = null;
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

    public override void BackpropagateNext(Matrix previouseJacobian)
    {
        _currentJacobian = previouseJacobian;
    }

    public override Vector CalculateValue() => Value;

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);
}