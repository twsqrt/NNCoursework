using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Activation;

public class ActivationNode : Node<Vector>
{
    private readonly Node<Vector> _child;
    private readonly Func<float, float> _function;
    private readonly Func<float, float> _derivative;
    private readonly ActivationType _type;

    private ActivationNode(Node<Vector> child, Func<float, float> function, Func<float, float> derivative, ActivationType type) 
    : base(child.Shape, new INode[]{child})
    {
        _child = child;
        _function = function;
        _derivative = derivative;
        _type = type;

        _value = Vector.CreateZero(Shape.Dimension);
        ParentGradient = Vector.CreateZero(Shape.Dimension);
    }

    public static ActivationNode Create(Node<Vector> child, ActivationType type)
    => type switch
    {
        ActivationType.PURELIN => new ActivationNode(child, x => x, x => 1.0f, type),
        ActivationType.SATLINS => new ActivationNode(
            child,
            ActivationFunctions.SymmetricSaturatingLinear,
            x => x > 0.0f && x < 1.0f ? 1.0f : 0.0f, 
            type
        ),
        ActivationType.RELU => new ActivationNode(
            child, 
            x => MathF.Max(0.01f * x, x),
            x => x > 0.0f ? 1.0f : 0.01f,
            type
        ),
        ActivationType.LOGSIG => new ActivationNode(
            child,
            ActivationFunctions.LogSigmoid,
            ActivationFunctions.LogSigmoidDerivative,
            type
        ),
        ActivationType.CUSTOM => throw new ArgumentException(),
        _  => throw new NotImplementedException(),
    };

    public static ActivationNode CreateCustorm(Node<Vector> child, Func<float, float> function, Func<float, float> derivative)
        => new ActivationNode(child, function, derivative, ActivationType.CUSTOM);

    public override void CalculateGradient()
    {
        for(int i = 0; i < ParentGradient.Dimension; i++)
            _child.ParentGradient[i] = ParentGradient[i] * _derivative(_child.Value[i]);
    }

    public override void CalculateValue()
    {
        for(int i = 0; i < _child.Value.Dimension; i++)
            _value[i] = _function(_child.Value[i]);
    }
}
