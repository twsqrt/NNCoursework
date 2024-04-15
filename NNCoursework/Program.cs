using LinearAlgebra;

var mat = Matrix.CreateFromFunction(4, 4, (i, j) => Math.Min(i + 1, j + 1));
var mat2 = Matrix.IdentityMatrix(5, 4);
Console.WriteLine(mat);
Console.WriteLine(mat2);