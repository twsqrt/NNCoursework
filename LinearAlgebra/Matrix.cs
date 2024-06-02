namespace LinearAlgebra;

public class Matrix : ITensor, IMatrix
{
    private readonly int _height;
    private readonly int _width;
    private readonly float[] _data;

    public float[] Data => _data;
    public int Height => _height;
    public int Width => _width;
    public (int, int) Shape => (_height, _width);

    public float this[int i, int j] 
    {
        get => _data[i * _width + j];
        set => _data[i * _width + j] = value;
    }

    public Matrix(float[] data, int height, int width)
    {
        if(data.Length != height * width)
            throw new ArgumentException();
        
        _height = height;
        _width = width;
        _data = data;
    }

    public void Add(Matrix other)
    {
        for(int i = 0; i < _height * _width; i++)
            _data[i] += other._data[i];
    }

    public static void Multiply(IMatrix lhs, IMatrix rhs, IMatrix result)
    {
        for(int i = 0; i < lhs.Height; i++)
        {
            for(int j = 0; j < rhs.Width; j++)
            {
                float sum = 0.0f;
                for(int k = 0; k < lhs.Width; k++)
                    sum += lhs[i, k] * rhs[k, j];
                
                result[i, j] = sum;
            }
        }
    }

    public static void Multiply(IMatrix lhs, Vector rhs, Vector result)
    {
        for(int i = 0; i < lhs.Height; i++)
        {
            float sum = 0.0f;
            for(int j = 0; j < rhs.Dimension;  j++)
                sum += rhs[j] * lhs[i, j];

            result[i] = sum;
        }
    }

    public static void Multiply(Vector lhs, IMatrix rhs, Vector result)
    {
        for(int i = 0; i < rhs.Height; i++)
        {
            float sum = 0.0f;
            for(int j = 0; j < lhs.Dimension;  j++)
                sum += lhs[j] * rhs[j, i];

            result[i] = sum;
        }
    }

    public static void Convolution(IMatrix matrix, IMatrix kernel, IMatrix result)
    {
        for(int i = 0; i < result.Height; i++)
        for(int j = 0; j < result.Width; j++)
        {
            float sum = 0.0f;

            for(int k = 0; k < kernel.Height; k++)
            for(int l = 0; l < kernel.Width; l++)
                sum += kernel[k, l] * matrix[i + k, j + l];
            
            result[i, j] = sum;
        }
    }

    public static void MaxPool(IMatrix matrix, int kernelHeight, int kernelWidth, IMatrix result)
    {
        for(int i = 0; i < result.Height; i++)
        for(int j = 0; j < result.Width; j++)
        {
            float max = matrix[i, j];

            for(int k = 0; k < kernelHeight; k++)
            for(int l = 0; l < kernelWidth; l++)
            {
                float element = matrix[i + k, j + l];
                if(max < element)
                    max = element;
            }
            
            result[i, j] = max;
        }

    }
}