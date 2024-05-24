using NeuralNetworks.Transfer;

internal static class ActivationFunctions
{
    public static float LogSigmoid(float x)
        => 1.0f / (1.0f + MathF.Exp(-x));
    
    public static float LogSigmoidDerivative(float x)
    {
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