using LinearAlgebra;

var mat1 = new Matrix(new double[,]{
    {1, 2, 3},
    {2, 1, 1},
});

var mat2 = new Matrix(new double[,]{
    {1, 2},
    {2, 2},
    {3, 3},
});

Console.WriteLine(mat1);
Console.WriteLine(mat2);

var vec = new Vector(new double[]{1, 2, 3});
var res = mat2 * mat1 * vec;
Console.WriteLine(res);