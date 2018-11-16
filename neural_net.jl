using DelimitedFiles
using Random
using Statistics

function sigmoid(x)
        return 1/(1+exp(-x));
end

function sigmoidDerivative(x)
        return sigmoid(x)*(1 - sigmoid(x));
end

data = readdlm(ARGS[1], ',', Float64, '\n');
numOfDatapoints = size(data,2)
hiddenLayerSize = 15;
learningRate = 0.2;

resistanceMean = mean(data[1,:]);
resistanceStdDev = std(data[1,:]);
capacitanceMean = mean(data[2,:]);
capacitanceStdDev = std(data[2,:]);
timeMean = mean(data[3,:]);
timeStdDev = std(data[3,:]);

inputs = vcat(data[1:3,:], ones(numOfDatapoints)'); # Adds row of ones for bias term in weights
expectedOutput = data[4,:];

# Normalize inputs
inputs[1,:] .-= resistanceMean;
inputs[1,:] ./= resistanceStdDev;
inputs[2,:] .-= capacitanceMean;
inputs[2,:] ./= capacitanceStdDev;
inputs[3,:] .-= timeMean;
inputs[3,:] ./= timeStdDev;

global layerOneWeights = rand!(zeros(hiddenLayerSize,4)); # Contains biases for efficiency
global layerTwoWeights = rand!(zeros(hiddenLayerSize + 1)'); # Also contains biases

errorPlotData = open("cost.csv", "a");

for j = 1:typemax(UInt64) # Effectively infinite
        layerOneGradient = zeros(hiddenLayerSize,4);
        layerTwoGradient = zeros(hiddenLayerSize + 1)';

        error = 0;

        for i = 1:numOfDatapoints
                netOne = layerOneWeights*inputs[:,i];
                layerOneOutput = vcat(sigmoid.(netOne), 1); # Append 1 for bias term in weights

                netTwo = layerTwoWeights*layerOneOutput;
                layerTwoOutput = netTwo; # Identity output unit.

                cost = (layerTwoOutput - expectedOutput[i])^2;
                error += cost;
                dCost_dY = 2*(layerTwoOutput - expectedOutput[i]);
                
                layerOneGradient[:,:] += dCost_dY * layerTwoWeights[1:hiddenLayerSize] .* sigmoidDerivative.(netOne) .* inputs[:,i]';

                layerTwoGradient[:] += dCost_dY * layerOneOutput[:];
        end
        
        layerOneGradient /= numOfDatapoints;
        layerTwoGradient /= numOfDatapoints;

        global layerOneWeights -= learningRate * layerOneGradient;
        global layerTwoWeights -= learningRate * layerTwoGradient;

        print("Error: ", error/numOfDatapoints, "\n", "Num of iterations: ", j, '\n')

        if mod(j,50) == 1
                write(errorPlotData, string(j), ",", string(error/numOfDatapoints), "\n");
        end

        if mod(j,1000) == 0
                writedlm("layer1.csv", layerOneWeights, ',');
                writedlm("layer2.csv", layerTwoWeights, ',');
        end
end
