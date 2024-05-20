using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NNCoursework.Tests;

[TestClass]
public class LayerTests
{
    [TestMethod]
    public void LayerOperation_OneLayer_CorrectValue()
    {
        float w11 = 1.0f;
        float w12 = 2.0f;
        float w21 = -1.0f;
        float w22 = 1.5f;

        var parameterValue = new Vector<float>(new float[] { 1.0f, 2.0f});
        var parameter = new Parameter(parameterValue);

        var weightsValue = new Vector<float>(new float[]{w11, w12, w21, w22});
        var weights = new Parameter(weightsValue);

        var layer = new LayerOperation(weights, parameter);
        var opB = new TestUnaryOperationB(layer);
        var opC = new TestUnaryOperationC(opB);

        opC.UpdateValue();

        float correctValue = MathF.Sqrt(w11 + 2.0f * w12) + w21 + 2.0f * w22 + MathF.Sin(w21 + 2.0f * w22) + 2.0f;
        Assert.AreEqual(correctValue, opC.CurrentValue.ToNumber(), 0.0001f);
    }

    [TestMethod]
    public void LayerOperation_OneLayer_CorrectJacobian()
    {
        float w11 = 1.0f;
        float w12 = 2.0f;
        float w21 = -1.0f;
        float w22 = 1.5f;

        var parameterValue = new Vector<float>(new float[] { 1.0f, 2.0f});
        var parameter = new Parameter(parameterValue);

        var weightsValue = new Vector<float>(new float[]{w11, w12, w21, w22});
        var weights = new Parameter(weightsValue);

        var layer = new LayerOperation(weights, parameter);
        var opB = new TestUnaryOperationB(layer);
        var opC = new TestUnaryOperationC(opB);

        opC.UpdateValue();
        opC.BackpropagateNext(Matrix<float>.CreateIdentityMatrix(1));

        float dw11 = weights.CurrentJacobian[0, 0];
        float dw12 = weights.CurrentJacobian[0, 1];
        float dw21 = weights.CurrentJacobian[0, 2];
        float dw22 = weights.CurrentJacobian[0, 3];

        float correctDw11 = 0.5f / MathF.Sqrt(w11 + 2.0f * w12);
        float correctDw12 = 2.0f * correctDw11;
        float correctDw21 = 1.0f + MathF.Cos(w21 + 2.0f * w22);
        float correctDw22 = 2.0f * correctDw21;

        Assert.AreEqual(correctDw11, dw11, 0.0001f);
        Assert.AreEqual(correctDw12, dw12, 0.0001f);
        Assert.AreEqual(correctDw21, dw21, 0.0001f);
        Assert.AreEqual(correctDw22, dw22, 0.0001f);
    }
}