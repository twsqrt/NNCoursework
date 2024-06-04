namespace LinearAlgebra;

public enum TensorType
{
    VECTOR,
    MATRIX,
    TENSOR3D
}

public interface ITensor
{
    TensorType Type { get; }
    float[] Data { get;}
}
