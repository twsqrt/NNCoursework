using System.Diagnostics;
using LinearAlgebra;
using NeuralNetworks.Transfer;

var mat1 = new Matrix<double>(new double[,]{
    {1, 2, 3},
    {2, 1, 1},
});

var mat2 = new Matrix<double>(new double[,]{
    {1, 2},
    {2, 2},
    {3, 3},
});

Console.WriteLine(mat1);
Console.WriteLine(mat2);

var vec = new Vector<double>(new double[]{-1, -2, 3});
var res = mat2 * mat1 * vec;
TransferFunction transfer = TransferFunction.Create(TransferFunctionType.LOGSIG);
Console.WriteLine("res: " + res);
Console.WriteLine("trasfer res: " + transfer.Execute(res));