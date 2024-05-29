using NeuralNetworks.ComputationGraph;
using NeuralNetworks.ComputationGraph.File;
using NeuralNetworks.Network;

namespace NeuralNetworks.File;

public static class NetworkFileManager
{
    public static void Write(NeuralNetwork network, BinaryWriter writer)
    {
        writer.Write(network.Input.ID);

        writer.Write(network.Parameters.Length);
        foreach(int parameterID in network.Parameters.Select(p => p.ID))
            writer.Write(parameterID);
        
        var gw = new GraphWriter(writer);
        network.Output.Accept(gw);
    }

    public static NeuralNetwork Read(BinaryReader reader)
    {
        int inputID = reader.ReadInt32();

        int numberOfParameters = reader.ReadInt32();
        var parametersID = new int[numberOfParameters];
        for(int i = 0; i < numberOfParameters; i++)
            parametersID[i] = reader.ReadInt32();

        ParameterNode input = null;
        var parameters = new List<ParameterNode>();

        var gr = new GraphReader(reader, (id, type, node) => {
            if(node is ParameterNode parameter)
            {
                if(parametersID.Contains(id))
                    parameters.Add(parameter);
                else if(inputID == id)
                    input = parameter;
            }
        });

        Node output = gr.ReadGraph();
        return new NeuralNetwork(input, parameters.ToArray(), output);
    }
}
