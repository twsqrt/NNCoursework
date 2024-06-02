namespace LinearAlgebra;

public class Vector : ITensor
{
    private readonly int _dimension;
    private readonly float[] _data;

    public int Dimension => _dimension;
    public float[] Data => _data;

    public float LengthSquared
    {
        get 
        {
            float sum = 0.0f;
            for(int i = 0; i < _dimension; i++)
                sum += _data[i] * _data[i];

            return sum;
        }
    }

    public float this[int i]
    {
        get => _data[i];
        set => _data[i] = value;
    }

    public Vector(float[] data)
    {
        _dimension = data.Length;
        _data = data;
    }

    public static Vector CreateZero(int length)
    {
        var data = new float[length];
        Array.Fill(data, 0.0f);
        return new Vector(data);
    }
    
    public static Vector Create1D(float value)
    {
        var data = new float[]{ value };
        return new Vector(data);
    }

    public void Add(Vector vector)
    {
        for(int i = 0; i < _dimension; i++)
            _data[i] += vector._data[i];
    }

    public void Scale(float value)
    {
        for(int i = 0; i < _dimension; i++)
            _data[i] *= value;
    }
    
    public void CopyValuesFrom(Vector value)
    {
        for(int i = 0; i < _data.Length; i++)
            _data[i] = value._data[i];
    }

    public float ToNumber()
    {
        if(_dimension != 1)
            throw new InvalidOperationException();
          
        return _data[0];
    }

    public static void Addition(Vector lhs, Vector rhs, Vector result)
    {
        for(int i = 0; i < result._data.Length; i++)
            result._data[i] = lhs._data[i] + rhs._data[i];
    }
    
    public static void Difference(Vector lhs, Vector rhs, Vector result)
    {
        for(int i = 0; i < result._data.Length; i++)
            result._data[i] = lhs._data[i] - rhs._data[i];
    }
}
