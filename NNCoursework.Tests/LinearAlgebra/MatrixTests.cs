namespace LinearAlgebra.Tests;

[TestClass]
public class LinearAlgebraTests
{
    [TestMethod]
    public void MatrixMultiplication_DefaultFloatMatrix_CorrectValue()
    {
        var mat1 = new Matrix<float>(3, 4, new float[] {-1, 0.5f, 2, 0.22f, 1, 0, 3, -1, 1.001f, -9999, 0.001f, -2});
        var mat2 = new Matrix<float>(4, 5, new float[] {9999, 0.001f, 0.5f, 0.23f, 0.1f, 1, 0, -1, 5, 0, 0.1234f, -0.001f, 1, -9999, 0, 1, 1.5f, 3, 2, 0});
        var correctResult = new Matrix<float>(3, 5, new float[] {-9998.0332f, 0.327f, 1.66f, -19995.29f, -0.1f, 9998.3702f, -1.502f, 0.5f, -29998.77f, 0.1f, 7.9991234f, -2.999f, 9993.5015f, -50008.76877f, 0.1001f});

        Matrix<float> result = mat2.MultiplyRight(mat1);
        float mse = 0.0f;
        for(int i = 0; i < result.Height; i++)
        {
            for(int j = 0; j < result.Width; j++)
            {
                float diff = correctResult[i, j] - result[i, j];
                mse += diff * diff;
            }
        }
        mse /= result.Width * result.Height;
        mse = MathF.Sqrt(mse);

        Assert.IsTrue(mse < 0.001f);
    }

    [TestMethod]
    public void MatrixCachedMultiplication_DefaultFloatMatrix_CorrectValue()
    {
        var mat1 = new Matrix<float>(3, 4, new float[] {-1, 0.5f, 2, 0.22f, 1, 0, 3, -1, 1.001f, -9999, 0.001f, -2});
        var mat2 = new Matrix<float>(4, 5, new float[] {9999, 0.001f, 0.5f, 0.23f, 0.1f, 1, 0, -1, 5, 0, 0.1234f, -0.001f, 1, -9999, 0, 1, 1.5f, 3, 2, 0});
        var correctResult = new Matrix<float>(3, 5, new float[] {-9998.0332f, 0.327f, 1.66f, -19995.29f, -0.1f, 9998.3702f, -1.502f, 0.5f, -29998.77f, 0.1f, 7.9991234f, -2.999f, 9993.5015f, -50008.76877f, 0.1001f});

        Matrix<float> result = mat2.MultiplyRightCached(mat1);
        float mse = 0.0f;
        for(int i = 0; i < result.Height; i++)
        {
            for(int j = 0; j < result.Width; j++)
            {
                float diff = correctResult[i, j] - result[i, j];
                mse += diff * diff;
            }
        }
        mse /= result.Width * result.Height;
        mse = MathF.Sqrt(mse);

        Assert.IsTrue(mse < 0.001f);
    }

    [TestMethod]
    public void MatrixCachedMultiplication_1DLeftFloatMatrix_CorrectValue()
    {
        var mat1 = new Matrix<float>(1, 4, new float[] {-1, 0.5f, 2, 0.22f});
        var mat2 = new Matrix<float>(4, 5, new float[] {9999, 0.001f, 0.5f, 0.23f, 0.1f, 1, 0, -1, 5, 0, 0.1234f, -0.001f, 1, -9999, 0, 1, 1.5f, 3, 2, 0});
        var correctResult = new Matrix<float>(1, 5, new float[] {-9998.0332f, 0.327f, 1.66f, -19995.29f, -0.1f});

        Matrix<float> result = mat2.MultiplyRight(mat1);
        float mse = 0.0f;
        for(int i = 0; i < result.Height; i++)
        {
            for(int j = 0; j < result.Width; j++)
            {
                float diff = correctResult[i, j] - result[i, j];
                mse += diff * diff;
            }
        }
        mse /= result.Width * result.Height;
        mse = MathF.Sqrt(mse);

        Assert.IsTrue(mse < 0.001f);
    }

    [TestMethod]
    public void MatrixCachedMultiplication_SquareFloatMatrix_CorrectValue()
    {
        var mat1 = new Matrix<float>(3, 4, new float[] {-1, 0.5f, 2, 0.22f, 1, 0, 3, -1, 1.001f, -9999, 0.001f, -2});
        var mat2 = new Matrix<float>(4, 4, new float[] {9999, 0.001f, 0.5f, 0.23f, 1, 0, -1, 5, 0.1234f, -0.001f, 1, -9999, 1, 1.5f, 3, 2});
        var correctResult = new Matrix<float>(3, 4, new float[] {-9998.0332f, 0.327f, 1.66f, -19995.29f, 9998.3702f, -1.502f, 0.5f, -29998.77f, 7.9991234f, -2.999f, 9993.5015f, -50008.76877f});

        Matrix<float> result = mat2.MultiplyRightCached(mat1);
        float mse = 0.0f;
        for(int i = 0; i < result.Height; i++)
        {
            for(int j = 0; j < result.Width; j++)
            {
                float diff = correctResult[i, j] - result[i, j];
                mse += diff * diff;
            }
        }
        mse /= result.Width * result.Height;
        mse = MathF.Sqrt(mse);

        Assert.IsTrue(mse < 0.001f);
    }

    [TestMethod]
    public void MatrixCachedMultiplication_DiagonalFloatMatrix_CorrectValue()
    {
        var mat1 = new Matrix<float>(3, 4, new float[] {-1, 0.5f, 2, 0.22f, 1, 0, 3, -1, 1.001f, -9999, 0.001f, -2});
        var mat2 = new DiagonalMatrix<float>(new float[] {0.12f, 1, -2.23f, 9999f});
        var correctResult = new Matrix<float>(3, 4, new float[] {-0.12f, 0.5f, -4.46f, 2199.78f, 0.12f, 0, -6.69f, -9999f, 0.12012f, -9999, -0.00223f, -19998f});

        Matrix<float> result = mat2.MultiplyRightCached(mat1);
        float mse = 0.0f;
        for(int i = 0; i < result.Height; i++)
        {
            for(int j = 0; j < result.Width; j++)
            {
                float diff = correctResult[i, j] - result[i, j];
                mse += diff * diff;
            }
        }
        mse /= result.Width * result.Height;
        mse = MathF.Sqrt(mse);

        Assert.IsTrue(mse < 0.001f);
    }
}