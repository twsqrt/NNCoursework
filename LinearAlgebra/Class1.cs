namespace LinearAlgebra;

public static class TestClass
{
    public static string TestFunction() => "Hello wolrd!";

    public static void TestFunctionTwo(string? line)
    {
        if(line is not null)
            Console.WriteLine("line: " + line);
    }
}
