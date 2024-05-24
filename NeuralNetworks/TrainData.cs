using LinearAlgebra;

namespace NeuralNetworks;

public readonly struct TrainData
{
    public Vector<float> Data {get; init;}
    public Vector<float> Markup {get; init;}

    public TrainData(float[] data, float[] markup)
    {
        Data = new Vector<float>(data);
        Markup = new Vector<float>(markup);
    }
}
