using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace ComputationGraph.Tests;

[TestClass]
public class OperationTests
{
    [TestMethod]
    public void UnaryOperations_ManyOperation_CorrectValue()
    {
        float x = 1.0f;
        float y = 2.0f;

        ParameterNode parameter = ParameterNode.CreateFromArray(new float[] {x, y});

        var opA = new TestUnaryOperationA(parameter);
        var opB = new TestUnaryOperationB(opA);
        var opC = new TestUnaryOperationC(opB);

        var result = opC.CalculateValue().ToNumber();

        float correctValue = MathF.Sqrt(x * x + y * y) + x * y + MathF.Sin(x * y + 1.0f) + 3.0f;
        Assert.AreEqual(correctValue, result, 0.0001f);
    }

    [TestMethod]
    public void UnaryOperations_ManyOperation_CorrectJacobian()
    {
        float x = 1.0f;
        float y = 2.0f;

        ParameterNode parameter = ParameterNode.CreateFromArray(new float[] {x, y});

        var opA = new TestUnaryOperationA(parameter);
        var opB = new TestUnaryOperationB(opA);
        var opC = new TestUnaryOperationC(opB);

        opC.CalculateValue();
        opC.Backpropagate();

        float correctDx = y + y * MathF.Cos(x * y + 1.0f) + x / MathF.Sqrt(x * x + y * y);
        float correctDy = x + x * MathF.Cos(x * y + 1.0f) + y / MathF.Sqrt(x * x + y * y);

        float dx = parameter.CurrentJacobian[0, 0];
        float dy = parameter.CurrentJacobian[0, 1];

        Assert.AreEqual(correctDx, dx, 0.0001f);
        Assert.AreEqual(correctDy, dy, 0.0001f);
    }

    [TestMethod]
    public void BinaryOperations_OneAdditionOperation_CorrectValue()
    {
        float x1 = 1.0f;
        float x2 = 2.0f;
        float y1 = 1.5f;
        float y2 = 0.5f;

        ParameterNode x = ParameterNode.CreateFromArray(new float[] {x1, x2});
        ParameterNode y = ParameterNode.CreateFromArray(new float[] {y1, y2});

        var opA = new TestUnaryOperationA(y);
        var add = new AdditionNode(opA, x, 1);
        var opB = new TestUnaryOperationB(add);
        var opC = new TestUnaryOperationC(opB);

        var result = opC.CalculateValue().ToNumber();
        
        float correctValue = MathF.Sqrt(y1 * y1 + y2 * y2 + x1) + y1 * y2 + x2 + MathF.Sin(y1 * y2 + x2 + 1) + 3;
        Assert.AreEqual(correctValue, result, 0.0001f);
    }

    [TestMethod]
    public void BinaryOperations_OneAdditionOperation_CorrectJacobian()
    {
        float x1 = 1.0f;
        float x2 = 2.0f;
        float y1 = 1.5f;
        float y2 = 0.5f;

        ParameterNode x = ParameterNode.CreateFromArray(new float[] {x1, x2});
        ParameterNode y = ParameterNode.CreateFromArray(new float[] {y1, y2});

        var opA = new TestUnaryOperationA(y);
        var add = new AdditionNode(opA, x, 1);
        var opB = new TestUnaryOperationB(add);
        var opC = new TestUnaryOperationC(opB);

        opC.CalculateValue();
        opC.Backpropagate();

        float dx1 = x.CurrentJacobian[0, 0];
        float dx2 = x.CurrentJacobian[0, 1];
        float dy1 = y.CurrentJacobian[0, 0];
        float dy2 = y.CurrentJacobian[0, 1];

        float correctDx1 = 0.5f / MathF.Sqrt(x1 + y1 * y1 + y2 * y2);
        float correctDx2 = MathF.Cos(x2 + y1 * y2 + 1.0f) + 1.0f;
        float correctDy1 = y1 / MathF.Sqrt(x1 + y2 * y2 + y1 * y1 ) + y2 * MathF.Cos(x2 + y1 * y2 + 1.0f) + y2;
        float correctDy2 = y2 / MathF.Sqrt(x1 + y2 * y2 + y1 * y1 ) + y1 * MathF.Cos(x2 + y1 * y2 + 1.0f) + y1;

        Assert.AreEqual(correctDx1, dx1, 0.0001f);
        Assert.AreEqual(correctDx2, dx2, 0.0001f);
        Assert.AreEqual(correctDy1, dy1, 0.0001f);
        Assert.AreEqual(correctDy2, dy2, 0.0001f);
    }

    [TestMethod]
    public void BinaryOperations_OneCustomOperation_CorrectValue()
    {
        float y1 = 0.5f;
        float y2 = 0.25f;
        float y3 = 2.0f;
        float x1 = 1.0f;

        ParameterNode x = ParameterNode.CreateFromArray(new float[] {x1});
        ParameterNode y = ParameterNode.CreateFromArray(new float[] {y1, y2, y3});

        var opD = new TestBinaryOperationD(y, x, 1);
        var opC = new TestUnaryOperationC(opD);

        var result = opC.CalculateValue().ToNumber();

        float correctValue = y1 * x1 + y2 * y3 + 1.0f * x1 + 1.0f / (y1 + x1) + MathF.Sqrt(y2 * y2 + x1 + y3) + 1.0f;
        Assert.AreEqual(correctValue, result, 0.0001f);
    }

    [TestMethod]
    public void BinaryOperations_OneCustomOperation_CorrectJacobian()
    {
        float y1 = 0.5f;
        float y2 = 0.25f;
        float y3 = 2.0f;
        float x1 = 1.0f;

        ParameterNode x = ParameterNode.CreateFromArray(new float[] {x1});
        ParameterNode y = ParameterNode.CreateFromArray(new float[] {y1, y2, y3});

        var opD = new TestBinaryOperationD(y, x, 1);
        var opC = new TestUnaryOperationC(opD);

        opC.CalculateValue();
        opC.Backpropagate();

        float dy1 = y.CurrentJacobian[0, 0];
        float dy2 = y.CurrentJacobian[0, 1];
        float dy3 = y.CurrentJacobian[0, 2];
        float dx = x.CurrentJacobian[0, 0];

        float correctDy1 = x1 - 1.0f / (y1 + x1) / (y1 + x1); 
        float correctDy2 = y2 / MathF.Sqrt(y2 * y2 + y3 + x1) + y3;
        float correctDy3 = 0.5f / MathF.Sqrt(y2 * y2 + y3 + x1) + y2;
        float correctDx = -1.0f / (y1 + x1) / (y1 + x1) + y1 + 0.5f / MathF.Sqrt(y2 * y2 + y3 + x1) - 1.0f / x1 / x1;

        Assert.AreEqual(correctDy1, dy1, 0.0001f);
        Assert.AreEqual(correctDy2, dy2, 0.0001f);
        Assert.AreEqual(correctDy3, dy3, 0.0001f);
        Assert.AreEqual(correctDx, dx, 0.0001f);
    }
}