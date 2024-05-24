using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;
public abstract class UnaryOperationNode : Node
{
    private readonly Node _child;
    private Vector<float> _childValue;

    public UnaryOperationNode(Node child, int dimension) : base(dimension)
    {
        _child = child;
        _childValue = null;
    }

    protected abstract Vector<float> Function(Vector<float> paraemter);
    protected abstract Matrix<float> GetJacobian(Vector<float> parameter);

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        Matrix<float> jacobian = GetJacobian(_childValue);

        _child.BackpropagateNext(jacobian.MultiplyRightCached(previouseJacobian));
    }

    public override Vector<float> CalculateValue()
    {
        _childValue = _child.CalculateValue();
        return Function(_childValue);
    }
}
