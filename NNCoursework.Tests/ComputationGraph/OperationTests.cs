using System.CodeDom.Compiler;
using System.Security.AccessControl;
using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NNCoursework.Tests;

[TestClass]
public class OperationTests
{
    [TestMethod]
    public void UnaryOperations_ManyOperation_CorrectValue()
    {
        float x = 1.0f;
        float y = 2.0f;

        var parameter = Parameter.CreateZero(2);
        parameter.SetValue(new Vector<float>(new float[] {x, y}));

        var opA = new TestUnaryOperationA(parameter);
        var opB = new TestUnaryOperationB(opA);
        var opC = new TestUnaryOperationC(opB);

        float correctValue = MathF.Sqrt(x * x + y * y) + x * y + MathF.Sin(x * y + 1.0f) + 3.0f;
        opC.UpdateValue();

        float diff = MathF.Abs(correctValue - opC.CurrentValue.ToNumber());
        Assert.IsTrue(diff < 0.0001f);
    }

    [TestMethod]
    public void UnaryOperations_ManyOperation_CorrectJacobian()
    {
        float x = 1.0f;
        float y = 2.0f;

        var parameter = Parameter.CreateZero(2);
        parameter.SetValue(new Vector<float>(new float[] {x, y}));

        var opA = new TestUnaryOperationA(parameter);
        var opB = new TestUnaryOperationB(opA);
        var opC = new TestUnaryOperationC(opB);

        opC.UpdateValue();
        opC.BackpropagateNext(Matrix<float>.IdentityMatrix(1));

        float correctDx = y + y * MathF.Cos(x * y + 1.0f) + x / MathF.Sqrt(x * x + y * y);
        float correctDy = x + x * MathF.Cos(x * y + 1.0f) + y / MathF.Sqrt(x * x + y * y);

        float dx = parameter.CurrentJacobian[0, 0];
        float dy = parameter.CurrentJacobian[0, 1];

        float diffDx = MathF.Abs(dx - correctDx);
        float diffDy = MathF.Abs(dy - correctDy);
        Assert.IsTrue(diffDx < 0.0001f && diffDy < 0.0001f);
    }
}