using System.Numerics;
using System.Text;

namespace LinearAlgebra;

public class Matrix<T>
    where T : INumber<T>
{
    private readonly (int, int) _size;
    private readonly T[,] _matrix;

    public int Height => _size.Item1;
    public int Width => _size.Item2;
    public (int, int) Size => _size;

    public T this[int height, int width] 
    {
        get => _matrix[height, width];
        set => _matrix[height, width] = value;
    }

    private Matrix(int height, int width)
    {
        _size = (height, width);
        _matrix = new T[height, width];
    }

    private Matrix(int height, int width, T[][] matrix)
        : this(height, width)
    {
        for(int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
                _matrix[i,j] = matrix[i][j];
        }
    }

    public Matrix(T[,] matrix)
    {
        _size = (matrix.GetLength(0), matrix.GetLength(1));
        _matrix = matrix;
    }

    public static Matrix<T> CreateFromFunction(int height, int width, Func<int, int, T> function) 
    {
        T[][] matrix = Enumerable.Range(0, height)
            .Select(i => Enumerable.Range(0, width).Select(j => function(i, j)).ToArray())
            .ToArray();
        return new Matrix<T>(height, width, matrix);
    }

    public static Matrix<T> ZeroMatrix(int height, int width)
        => CreateFromFunction(height, width, (_, _) => T.Zero);

    public static Matrix<T> IdentityMatrix(int height, int width)
        => CreateFromFunction(height, width, (i, j) => i == j ? T.One : T.Zero);

    public override string ToString()
    {
        var sb = new StringBuilder();

        sb.AppendLine("[");
        for (int i = 0; i < Height; i++)
        {
            sb.Append("\t[");
            for(int j =0; j < Width; j++)
                sb.Append($"{_matrix[i, j]},\t");
            sb.AppendLine("]");
        }
        sb.Append("]");

        return sb.ToString();
    }

    public static Matrix<T> operator *(T scalar, Matrix<T> matrix)
        => CreateFromFunction(matrix.Height, matrix.Width, (i, j) => scalar * matrix[i, j]);
    
    public static Matrix<T> operator +(Matrix<T> matrix) => matrix;
    public static Matrix<T> operator -(Matrix<T> matrix) => - T.One * matrix;

    public static Matrix<T> operator +(Matrix<T> lhs, Matrix<T> rhs)
    {
        if(lhs.Size != rhs.Size)
            throw new ArgumentException();

        (int height, int width) = lhs.Size;
        return CreateFromFunction(height, width, (i, j) => lhs[i, j] + rhs[i, j]);
    }

    public static Matrix<T> operator -(Matrix<T> lhs, Matrix<T> rhs)
    {
        if(lhs.Size != rhs.Size)
            throw new ArgumentException();

        (int height, int width) = lhs.Size;
        return CreateFromFunction(height, width, (i, j) => lhs[i, j] - rhs[i, j]);
    }

    public static Matrix<T> operator *(Matrix<T> lhs, Matrix<T> rhs)
    {
        if(lhs.Width != rhs.Height)
            throw new ArgumentException();

        int length = lhs.Width;
           
        var result = new Matrix<T>(lhs.Height, rhs.Width);
        for(int i = 0; i < result.Height; i++)
        {
            for(int j = 0; j < result.Width; j++)
            {
                T scalarProduct = T.Zero;
                for(int k = 0; k < length; k++)
                    scalarProduct += lhs[i, k] * rhs[k, j];
                result[i, j] = scalarProduct;
            }
        }

        return result;
    }

    public static Vector<T> operator *(Matrix<T> matrix, Vector<T> vector)
    {
        if(matrix.Width != vector.Length)
            throw new ArgumentException();
        
        int resultLength = matrix.Height;
        var result = new T[resultLength];
        for(int i =0; i < resultLength; i++)
        {
            T scalarProduct = T.Zero;
            for(int j =0; j < vector.Length; j++)
                scalarProduct += matrix[i, j] * vector[j];
            result[i] = scalarProduct;
        }

        return new Vector<T>(result);
    }
}