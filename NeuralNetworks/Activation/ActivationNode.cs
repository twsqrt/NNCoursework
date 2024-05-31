using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Activation;

public class ActivationNode : Node
{
    private readonly Node _child;
    private readonly Func<float, float> _function;
    private readonly Func<float, float> _derivative;
    private readonly ActivationType _type;
    private readonly Vector _cachedResult;
    private Vector _childResult;

    public Node Child => _child;
    public ActivationType Type => _type;
    
    private ActivationNode(Node child, Func<float, float> function, Func<float, float> derivative, ActivationType type) 
    : base(child.Dimension)
    {
        _child = child;
        _function = function;
        _derivative = derivative;
        _type = type;

        _cachedResult = Vector.CreateZero(Dimension);
    }

    public static ActivationNode Create(Node child, ActivationType type)
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

    public static ActivationNode CreateCustorm(Node child, Func<float, float> function, Func<float, float> derivative)
        => new ActivationNode(child, function, derivative, ActivationType.CUSTOM);

    public override void BackpropagateNext(Matrix previouseJacobian)
    {
        for(int j = 0; j < previouseJacobian.Width; j++)
        {
            float diagonalElement = _derivative(_childResult[j]);
            for(int i = 0; i < previouseJacobian.Height; i++)
                previouseJacobian[i, j] *= diagonalElement;
        }

        _child.BackpropagateNext(previouseJacobian);
    }

    public override Vector CalculateValue()
    {
        _childResult = _child.CalculateValue();
        for(int i = 0; i < _childResult.Dimension; i++)
            _cachedResult[i] = _function(_childResult[i]);
           
        return _cachedResult;
    }

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);
}
