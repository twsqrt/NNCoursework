using System.Reflection;

namespace LinearAlgebra;

public static class TensorFactory
{
    private static readonly Random _random = new Random();

    private static ITensor Create(float[] data, Type type, TensorShape3D shape)
    {
        if(type == typeof(Vector) && shape.Width == 1 && shape.Depth == 1)
            return new Vector(data);
        if(type == typeof(Matrix) && shape.Depth == 1)
            return new Matrix(data, shape.Height, shape.Width);
        if(type == typeof(Tensor3D))
            return new Tensor3D(data, shape);
        
        throw new InvalidOperationException();
    }

    public static T Create<T>(float[] data, TensorShape3D shape) where T : ITensor
        => (T) Create(data, typeof(T), shape);
    
    public static T CreateZero<T>(TensorShape3D shape) where T : ITensor
    {
        float[] data = Enumerable.Range(0, shape.Dimension)
            .Select(_ => 0.0f)
            .ToArray();
        
        return Create<T>(data, shape);
    }

    public static T CreateRandom<T>(TensorShape3D shape, float min = -1.0f, float max = 1.0f) where T : ITensor
    {
        float[] data = Enumerable.Range(0, shape.Dimension)
            .Select(_ => (float) _random.NextDouble() * (max - min) + min)
            .ToArray();
        
        return Create<T>(data, shape);
    }
}
