using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class DataNode : Node<Vector>
{
    private static readonly Random _random = new Random();

    private Vector _gradient;

    public new Vector Value
    {
        get => _value;
        set => _value = value;
    }

    public Vector Gradient => _gradient;

    public DataNode(int dimension)
    : base(new TensorShape(dimension), new INode[0]) {}
   
    public DataNode(Vector initData)
    : base(new TensorShape(initData.Dimension), new INode[0])
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

    public override void BackpropagateNext(Vector gradinet)
        => _gradient = gradinet;

    public override void CalculateValue() {}
}
