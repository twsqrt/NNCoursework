using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class AdditionNode : Node
{
    private readonly Node _lhs;
    private readonly Node _rhs;
    private readonly Matrix<float> _rhsCachedJacobian;
    private readonly Vector<float> _additionCachedResult;

    public AdditionNode(Node lhs, Node rhs, int graphRootDimension) : base(lhs.Dimension)
    {
        if(lhs.Dimension != rhs.Dimension)
            throw new ArgumentException();

        _lhs = lhs;
        _rhs = rhs;
        _rhsCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, Dimension);
        _additionCachedResult = Vector<float>.CreateZeroVector(Dimension);
    }

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        _rhsCachedJacobian.CopyValuesFrom(previouseJacobian);

        _lhs.BackpropagateNext(previouseJacobian);
        _rhs.BackpropagateNext(_rhsCachedJacobian);
    }

    public override Vector<float> CalculateValue()
    {
        Vector<float>.Addition(_lhs.CalculateValue(), _rhs.CalculateValue(), _additionCachedResult);
        return _additionCachedResult;
    }
}