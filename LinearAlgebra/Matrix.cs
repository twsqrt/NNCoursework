using System.Text;

namespace LinearAlgebra;

public class Matrix
{
    private readonly int _width;
    private readonly int _height;
    private readonly double[,] _matrix;

    public int Width => _width;
    public int Height => _height;

    public double this[int height, int width] 
    {
        get => _matrix[height, width];
        set => _matrix[height, width] = value;
    }

    private Matrix(int height, int width, double[][] matrix)
    {
        _height = height;
        _width = width;

        _matrix = new double[height, width];
        for(int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
                _matrix[i,j] = matrix[i][j];
        }
    }

    public Matrix(double[,] matrix)
    {
        _height = matrix.GetLength(0);
        _width = matrix.GetLength(1);

        _matrix = matrix;
    }

    public static Matrix CreateFromFunction(int height, int width, Func<int, int, double> function) 
    {
        double[][] matrix = Enumerable.Range(0, height)
            .Select(i => Enumerable.Range(0, width).Select(j => function(i, j)).ToArray())
            .ToArray();
        return new Matrix(height, width, matrix);
    }

    public static Matrix ZeroMatrix(int height, int width)
        => CreateFromFunction(height, width, (_, _) => 0.0);

    public static Matrix IdentityMatrix(int height, int width)
        => CreateFromFunction(height, width, (i, j) => i == j ? 1.0 : 0.0);

    public override string ToString()
    {
        var sb = new StringBuilder();

        sb.AppendLine("[");
        for (int i = 0; i < _height; i++)
        {
            sb.Append("\t[");
            for(int j =0; j < _width; j++)
                sb.Append($"{_matrix[i, j]}, ");
            sb.AppendLine("]");
        }
        sb.AppendLine("]");

        return sb.ToString();
    }
}