using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class DataNode : Node<Vector>
{
    private static readonly Random _random = new Random();

    public Vector Gradient => ParentGradient;
    public new Vector Value
    {
        get => _value;
        set => _value = value;
    }

    public DataNode(int dimension)
    : base(new TensorShape3D(dimension), new INode[0])
    {
        ParentGradient = Vector.CreateZero(dimension);
    }
   
    public DataNode(Vector initData)
    : this(initData.Dimension)
    {
        _value = initData;
    }

    public static DataNode CreateFromArray(float[] data)
        => new DataNode(new Vector(data));
    
    public static DataNode CreateRandom(int dimension, float min = -1.0f, float max = 1.0f)
    {
        float[] data = Enumerable.Range(0, dimension)
            .Select(_ => (float)_random.NextDouble() * (max - min) + min)
            .ToArray();
        
        return CreateFromArray(data);
    }

    public override void CalculateGradient() {}
    public override void CalculateValue() {}
}
