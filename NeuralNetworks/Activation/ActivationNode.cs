using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Activation;

public class ActivationNode<T> : Node<T>
    where T : ITensor
{
    private readonly Node<T> _child;
    private readonly Func<float, float> _function;
    private readonly Func<float, float> _derivative;
    private readonly ActivationType _type;

    private ActivationNode(Node<T> child, Func<float, float> function, Func<float, float> derivative, ActivationType type) 
    : base(child.Shape, new INode[]{child})
    {
        _child = child;
        _function = function;
        _derivative = derivative;
        _type = type;
    }

    public static ActivationNode<T> Create(Node<T> child, ActivationType type)
    => type switch
    {
        ActivationType.PURELIN => new ActivationNode<T>(child, x => x, x => 1.0f, type),
        ActivationType.SATLINS => new ActivationNode<T>(
            child,
            ActivationFunctions.SymmetricSaturatingLinear,
            x => x > 0.0f && x < 1.0f ? 1.0f : 0.0f, 
            type
        ),
        ActivationType.RELU => new ActivationNode<T>(
            child, 
            x => MathF.Max(0.01f * x, x),
            x => x > 0.0f ? 1.0f : 0.01f,
            type
        ),
        ActivationType.LOGSIG => new ActivationNode<T>(
            child,
            ActivationFunctions.LogSigmoid,
            ActivationFunctions.LogSigmoidDerivative,
            type
        ),
        ActivationType.CUSTOM => throw new ArgumentException(),
        _  => throw new NotImplementedException(),
    };

    public static ActivationNode<T> CreateCustorm(Node<T> child, Func<float, float> function, Func<float, float> derivative)
        => new ActivationNode<T>(child, function, derivative, ActivationType.CUSTOM);

    public override void CalculateGradient()
    {
        for(int i = 0; i < ParentGradient.Data.Length; i++)
            _child.ParentGradient.Data[i] = 
                ParentGradient.Data[i] * _derivative(_child.Value.Data[i]);
    }

    public override void CalculateValue()
    {
        for(int i = 0; i < _child.Value.Data.Length; i++)
            _value.Data[i] = _function(_child.Value.Data[i]);
    }
}
