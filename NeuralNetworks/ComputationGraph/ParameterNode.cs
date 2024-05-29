using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class ParameterNode : Node
{
    private static readonly Random _random = new Random();

    private Matrix<float> _currentJacobian;

    public Matrix<float> CurrentJacobian => _currentJacobian;

    public ParameterNode(Vector<float> value) : base(value.Dimension)
    {
        Value = value;

        _currentJacobian = null;
    }

    public static ParameterNode CreateZero(int dimension)
        => new ParameterNode(Vector<float>.CreateZeroVector(dimension));
    
    public static ParameterNode CreateFromArray(float[] data)
        => new ParameterNode(new Vector<float>(data));
    
    public static ParameterNode CreateRandom(int dimension, float cordMinValue = 0.0f, float cordMaxValue = 1.0f)
    {
        var data = new float[dimension];
        for(int i = 0; i < data.Length; i++)
        {
            float cord = (float) _random.NextDouble() * (cordMaxValue - cordMinValue) + cordMinValue;
            data[i] = cord;
        }

        return new ParameterNode(new Vector<float>(data));
    }

    public Vector<float> Value { get; set; }

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        _currentJacobian = previouseJacobian;
    }

    public override Vector<float> CalculateValue() => Value;

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);
}