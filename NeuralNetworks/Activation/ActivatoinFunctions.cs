namespace NeuralNetworks.Activation;

internal static class ActivationFunctions
{
    private const int DECIMAL_PERCISION = 6;
    private static readonly float _constantApproximationBorder = DECIMAL_PERCISION * MathF.Log(10.0f);
    public static float LogSigmoid(float x)
    {
        if(MathF.Abs(x) > _constantApproximationBorder)
            return x > 0.0f ? 1.0f : 0.0f;

        return 1.0f / (1.0f + MathF.Exp(-x));
    }
    
    public static float LogSigmoidDerivative(float x)
    {
        if(Math.Abs(x) > _constantApproximationBorder)
            return 0.0f;

        float expMinusX = MathF.Exp(-x);
        return expMinusX / (expMinusX + 1.0f) / (expMinusX + 1.0f);
    }

    public static float SymmetricSaturatingLinear(float x)
    {
        if(x > 1.0f)
            return 1.0f;
        else if(x < -1.0f)
            return -1.0f;
        return x;
    }
}