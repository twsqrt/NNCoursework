using System.Text;

namespace LinearAlgebra;

public class Vector
{
    private readonly int _length;
    private readonly double[] _vector;

    public int Length => _length;

    public double this[int index]
    {
        get => _vector[index];
        set => _vector[index] = value;
    }

    private Vector(int length)
    {
        _length = length;
        _vector = new double[length];
    }

    public Vector(double[] vector)
    {
        _length = vector.Length;
        _vector = vector;
    }

    public static Vector Zero(int length)
        => new Vector(Enumerable.Range(0, length).Select(_ => 0.0).ToArray());
    
    public static Vector CreateFromFunction(int length, Func<int, double> function)
    {
        var result = new Vector(length);
        for(int i =0; i < length; i++)
            result[i] = function(i);
          
        return result;
    }
    
    public static double ScalarProduct(Vector lhs, Vector rhs)
    {
        if(lhs.Length != rhs.Length)
            throw new ArgumentException();
        
        double result = 0.0;
        for(int i =0; i < lhs.Length; i++)
            result += lhs[i] * rhs[i];
        return result;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();

        sb.Append("[");
        for(int i =0; i < _length; i++)
            sb.Append($"{_vector[i]},\t");
        sb.AppendLine("]");

        return sb.ToString();
    }

    public static Vector operator *(double scalar, Vector vector)
        => CreateFromFunction(vector.Length, i => scalar * vector[i]);

    public static Vector operator +(Vector vector) => vector;
    public static Vector operator -(Vector vector) => -1.0 * vector;

    public static Vector operator +(Vector lhs, Vector rhs)
    {
        if(lhs.Length != rhs.Length)
            throw new ArgumentException();
        
        return CreateFromFunction(lhs.Length, i => lhs[i] + rhs[i]);
    }

    public static Vector operator -(Vector lhs, Vector rhs)
    {
        if(lhs.Length != rhs.Length)
            throw new ArgumentException();
        
        return CreateFromFunction(lhs.Length, i => lhs[i] - rhs[i]);
    }

    public static double operator *(Vector lhs, Vector rhs)
        => ScalarProduct(lhs, rhs);
}
