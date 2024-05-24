using LinearAlgebra;
using NeuralNetworks;
using NeuralNetworks.Network;
using NeuralNetworks.Transfer;

float f(float x) => MathF.Cos(3.0f * MathF.Exp(2.0f * x) * MathF.Cos(2.0f * x));

const int DATA_SIZE = 200;

var random = new Random();
var data = new Vector<float>[DATA_SIZE];
var markup = new Vector<float>[DATA_SIZE];

for(int i = 0; i < DATA_SIZE; i++)
{
    float x = i * 2.0f / DATA_SIZE -1.0f;
    float y = f(x);
    y += (float) random.NextDouble() * 0.2f - 0.1f;
    
    data[i] = Vector<float>.Create1DVector(x);
    markup[i] = Vector<float>.Create1DVector(y);
}

var builder = new NetworkBuilder();
Network network = builder.Create()
    .WithInput(1)
    .ToLayer(3)
    .WithActivationFunction(ActivationType.LOGSIG)
    .ToLayer(10)
    .WithActivationFunction(ActivationType.LOGSIG)
    .ToLayer(1)
    .ToOutput()
    .Build();

//var sgdMethod = new RegularSGD(0.05f);
var sgdMethod = new RMSprop(0.9f, 0.05f);

network.Fit(data, markup, 5, 300000, sgdMethod, Console.Out);

string docPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);

using (var output = new StreamWriter(Path.Combine(docPath, "input_data.csv")))
{
    for(int i = 0; i < DATA_SIZE; i++)
    {
        float x = data[i].ToNumber();
        float y = markup[i].ToNumber();

        output.WriteLine($"{x};{y}");
    }
}

using (var output = new StreamWriter(Path.Combine(docPath, "result.csv")))
{
    for(int i = 0; i < 1000; i++)
    {
        float x = i / 500.0f -1.0f;
        float y = network.Execute(Vector<float>.Create1DVector(x)).ToNumber();

        output.WriteLine($"{x};{y}");
    }
}