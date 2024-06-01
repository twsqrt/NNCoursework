using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class DataNode : Node<Vector>
{
    private static readonly Random _random = new Random();

    private Vector _gradient;

    public Vector Data { get; set; }
    public Vector Gradient => _gradient;

    public DataNode(int dimension)
    : base(new TensorShape(dimension)) {}
    
    public DataNode(Vector initData)
    : base(new TensorShape(initData.Dimension))
    {
        Data = initData;
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

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);

    public override void BackpropagateNext(Vector gradient)
        => _gradient = gradient;

    public override Vector CalculateValue()
        => Data;
}
