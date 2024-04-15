namespace NeuralNetworks.Transfer;

internal static class ActivationFunctions
{
    public static double SymmetricSaturatingLinear(double x)
    {
        if(x > 1.0)
            return 1.0;
        else if(x < -1.0)
            return -1.0;
        return x;
    }

    public static double LogSigmoid(double x)
    {
        return 1.0 / (1 + Math.Exp(-x));
    }

    public static double HyperbolicTangentSigmoid(double x)
    {
        double exp2X = Math.Exp(2.0 * x);
        return (exp2X - 1) / (exp2X + 1);
    }

    public static Func<double, double> Get(TransferFunctionType functionType) => functionType switch
    {
        TransferFunctionType.HARDLIM => x => x > 0.0 ? 1.0 : 0.0,
        TransferFunctionType.HARDLIMS => x => x > 0.0 ? 1.0 : -1.0,
        TransferFunctionType.PURELIN => x => x,
        TransferFunctionType.SATLINS => SymmetricSaturatingLinear,
        TransferFunctionType.POSLIN => x => x > 0.0 ? x : 0.0,
        TransferFunctionType.LOGSIG => LogSigmoid,
        TransferFunctionType.TANSIG => HyperbolicTangentSigmoid,
        _ => throw new NotImplementedException(),
    };

}