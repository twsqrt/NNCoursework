using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Activation;

public class ActivationNode : Node
{
    private readonly Func<float, float> _function;
    private readonly Func<float, float> _derivative;
    private readonly Node _child;
    private readonly float[] _diagonalElements;
    private Vector<float> _childValue;
    
    public ActivationNode(Node child, Func<float, float> function, Func<float, float> derivative) : base(child.Dimension)
    {
        _child = child;
        _function = function;
        _derivative = derivative;

        _childValue = Vector<float>.CreateZeroVector(_child.Dimension);
        _diagonalElements = new float[Dimension];
    }

    public static ActivationNode Create(Node child, ActivationType type)
    => type switch
    {
        ActivationType.PURELIN => new ActivationNode(child, x => x, x => 1.0f),
        ActivationType.SATLINS => new ActivationNode(
            child,
            ActivationFunctions.SymmetricSaturatingLinear,
            x => x > 0.0f && x < 1.0f ? 1.0f : 0.0f
        ),
        ActivationType.POSLIN => new ActivationNode(
            child, 
            x => x > 0.0f ? x : 0.0f,
            x => x > 0.0f ? 1.0f : 0.0f
        ),
        ActivationType.LOGSIG => new ActivationNode(
            child,
            ActivationFunctions.LogSigmoid,
            ActivationFunctions.LogSigmoidDerivative
        ),
        _  => throw new NotImplementedException(),
    };

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        for(int j = 0; j < previouseJacobian.Width; j++)
        {
            float diagonalElement =  _derivative(_childValue[j]);
            _diagonalElements[j] = diagonalElement;

            for(int i = 0; i < previouseJacobian.Height; i++)
                previouseJacobian[i, j] *= diagonalElement;
        }

        _child.BackpropagateNext(previouseJacobian);
    }

    public override Vector<float> CalculateValue()
    {
        _childValue = _child.CalculateValue();

        var data = new float[Dimension];
        for(int i = 0; i < data.Length; i++)
            data[i] = _function(_childValue[i]);
           
        return new Vector<float>(data);
    }
}
