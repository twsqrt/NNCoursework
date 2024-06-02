using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public static class DataNodeFactory
{
    private static readonly Random _random = new Random();

    private static float[] CreateRandomData(int size, float min, float max)
    {
        return Enumerable.Range(0, size)
            .Select(_ => (float)_random.NextDouble() * (max - min) + min)
            .ToArray();
    }

    public static DataNode<Vector> CreateRandomVector(int dimension, float min = -1.0f, float max = 1.0f)
    {
        float[] data = CreateRandomData(dimension, min, max);
        var wrapper = new Vector(data);
        return new DataNode<Vector>(data, 
            wrapper, 
            Vector.CreateZero(dimension), 
            new TensorShape3D(dimension));
    }

    public static DataNode<Matrix> CreateRandomMatrix(int height, int width, float min = -1.0f, float max = 1.0f)
    {
        float[] data = CreateRandomData(height * width, min, max);
        var wrapper = new Matrix(height, width, data);
        return new DataNode<Matrix>(data, 
            wrapper, 
            Matrix.CreateZero(height, width),
            new TensorShape3D(height, width));
    }

    public static DataNode<Tensor3D> CreateRandomTensor(TensorShape3D shape, float min = -1.0f, float max = 1.0f)
    {
        float[] data = CreateRandomData(shape.Dimension, min, max);
        var wrapper = new Tensor3D(data, shape);
        return new DataNode<Tensor3D>(data, 
            wrapper, 
            Tensor3D.CreateZero(shape),
            shape);
    }


}
