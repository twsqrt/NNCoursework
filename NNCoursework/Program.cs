using LinearAlgebra;
using LinearAlgebra;
using NeuralNetworks.Transfer;

var mat1 = new Matrix<double>(3, 3, new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
var diag = new DiagonalMatrix<double>(new double[] {1, 2, 3});

Console.WriteLine(mat1);
Console.WriteLine(diag);

var res1 = diag.MultiplyRight(mat1);
Console.WriteLine("result1: " + res1.ToString());

var res2 = mat1.MultiplyRight(diag);
Console.WriteLine("result2: " + res2.ToString());

var res3 = diag.MultiplyRightCached(mat1);
Console.WriteLine("result3" + res3.ToString());