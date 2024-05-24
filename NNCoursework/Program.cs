using LinearAlgebra;
using NeuralNetworks;
using NeuralNetworks.Network;
using NeuralNetworks.Transfer;

var builder = new NetworkBuilder();
Network network = builder.Create()
    .WithInput(1)
    .ToLayer(10)
    .WithTransferFunction(ActivationType.LOGSIG)
    .ToLayer(10)
    .WithTransferFunction(ActivationType.LOGSIG)
    .ToLayer(1)
    .WithTransferFunction(ActivationType.PURELIN)
    .ToOutput()
    .Build();



float f(float x) => MathF.Cos(3.0f * MathF.Exp(2.0f * x) * MathF.Cos(2.0f * x));
//float f(float x) => 2.0f * x -1.0f;

var random = new Random();
var data = new TrainData[200];
for(int i = 0; i < 200; i++)
{
    float x = i / 100.0f -1.0f;
    float y = f(x);

    y += (float) random.NextDouble() * 0.2f - 0.1f;
    data[i] = new TrainData(new float[]{x}, new float[]{y});
}

network.Fit(data, 2000000, 0.1f);

string docPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);

using (var output = new StreamWriter(Path.Combine(docPath, "input_data.csv")))
{
    foreach(TrainData row in data)
    {
        float x = row.Data.ToNumber();
        float y = row.Markup.ToNumber();

        output.WriteLine($"{x};{y}");
    }
}

using (var output = new StreamWriter(Path.Combine(docPath, "result.csv")))
{
    for(int i = 0; i < 1000; i++)
    {
        float x = i / 500.0f -1.0f;
        var vector = new Vector<float>(new float[] {x});

        float y = network.Execute(vector).ToNumber();

        output.WriteLine($"{x};{y}");
    }
}