namespace LinearAlgebra;

public readonly struct TensorShape
{
    public readonly int Height;
    public readonly int Width;
    public readonly int Depth;

    public int Dimension => Height * Width * Depth;

    public TensorShape(int height, int width = 1, int depth = 1)
    {
        Height = height;
        Width = width;
        Depth = depth;
    }

    public override bool Equals(object? obj)
    {
        if (obj == null || GetType() != obj.GetType())
            return false;
        
        TensorShape other = (TensorShape) obj;
        return Height == other.Height
            && Width == other.Width
            && Depth == other.Depth;
    }
    
    public override int GetHashCode() 
        => (Height, Width, Depth).GetHashCode();
    
    public static bool operator ==(TensorShape lhs, TensorShape rhs)
        => lhs.Equals(rhs);

    public static bool operator !=(TensorShape lhs, TensorShape rhs)
        => ! lhs.Equals(rhs);
}
