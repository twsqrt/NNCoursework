using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class RMSprop : ISGDMethod
{
    private readonly float _beta;
    private readonly float _eta;
    private float _v;

    public RMSprop(float beta, float eta)
    {
        _beta = beta;
        _eta = eta;

        _v = 0.0f;
    }

    public float CalculateLearningRate(ParameterNode[] parameter, Vector<float>[] gradient)
    {
        _v *= _beta;
        foreach(Vector<float> parameterGradient in gradient)
            _v += (1.0f - _beta) * parameterGradient.LengthSquared;
        
        return _eta / MathF.Sqrt(_v + 0.0001f);
    }
}
