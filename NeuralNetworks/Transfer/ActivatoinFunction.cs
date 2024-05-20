using NeuralNetworks.Transfer;

public struct ActivationFunction
{
    public Func<float, float> Function { get; init; }
    public Func<float, float> Derivative { get; init; }

    public ActivationFunction(Func<float, float> function, Func<float, float> derivative)
    {
        Function = function;
        Derivative = derivative;
    }

    private static float LogSigmoid(float x)
        => 1.0f / (1.0f + MathF.Exp(-x));
    
    private static float LogSigmoidDerivative(float x)
    {
        float expMinusX = MathF.Exp(-x);
        return expMinusX / (expMinusX + 1.0f) / (expMinusX + 1.0f);

    }

    private static float SymmetricSaturatingLinear(float x)
    {
        if(x > 1.0f)
            return 1.0f;
        else if(x < -1.0f)
            return -1.0f;
        return x;
    }

    public static ActivationFunction CreateSatLins()
        => new ActivationFunction(SymmetricSaturatingLinear, x => x < 1.0f && x > 0.0f ? 1.0f : 0.0f);
    
    public static ActivationFunction CreatePosLin()
        => new ActivationFunction(x => x > 0.0f ? x : 0.0f, x => x > 0.0f ? 1.0f : 0.0f);
    
    public static ActivationFunction CreateLogSigmoid()
        => new ActivationFunction(LogSigmoid, LogSigmoidDerivative);

    public static ActivationFunction Create(ActivationFunctionType functionType)
    => functionType switch
    {
        ActivationFunctionType.SATLINS => CreateSatLins(),
        ActivationFunctionType.POSLIN => CreatePosLin(),
        ActivationFunctionType.LOGSIG => CreateLogSigmoid(),
        _ => throw new NotImplementedException()
    };
}