namespace LinearAlgebra;

public interface IMatrix
{
    int Height { get; } 
    int Width { get;}

    float this[int i, int j] { get; set; }
}
