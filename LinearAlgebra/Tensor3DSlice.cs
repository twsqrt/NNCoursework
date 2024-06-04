
namespace LinearAlgebra;

public class Tensor3DSlice : IMatrix
{
    private readonly float[] _data;
    private readonly int _dataStartIndex;
    private readonly int _height;
    private readonly int _width;

    public int Height => _height;
    public int Width => _width;

    public float this[int i, int j]
    {
        get => _data[i * _width + j + _dataStartIndex];
        set => _data[i * _width + j + _dataStartIndex] = value;
    }

    public Tensor3DSlice(int height, int width, float[] data, int dataStartIndex)
    {
        _height = height;
        _width = width;
        _data = data;
        _dataStartIndex = dataStartIndex;
    }
}
