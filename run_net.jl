using DelimitedFiles

function sigmoid(x)
        return 1/(1+exp.(-x));
end

# Set manually so that scaling is equivalent to training set.
resistanceMean = 49999.784590407784;
resistanceStdDev = 28617.43200535356;
capacitanceMean = 4.894379617495234e-6;
capacitanceStdDev = 2.9095702774112495e-6;
timeMean = 0.6064870129571632;
timeStdDev = 0.7246955112804028;

layerOneWeights = readdlm("layer1.csv", ',', Float64, '\n');
layerTwoWeights = readdlm("layer2.csv", ',', Float64, '\n');

data = readdlm(ARGS[1], ',', Float64, '\n');
numOfTestPoints = size(data, 2);

inputs = vcat(data[1:3,:], ones(numOfTestPoints)'); # Adds row of ones for bias term in weights
expectedOutput = data[4,:];
print(size(expectedOutput))
 
# Normalize inputs
inputs[1,:] .-= resistanceMean;
inputs[1,:] ./= resistanceStdDev;
inputs[2,:] .-= capacitanceMean;
inputs[2,:] ./= capacitanceStdDev;
inputs[3,:] .-= timeMean;
inputs[3,:] ./= timeStdDev;

global cost = 0.0;

for i = 1:numOfTestPoints
        netOne = layerOneWeights * inputs[:,i];
        outputOne = vcat(sigmoid.(netOne), 1);
       
        netTwo = layerTwoWeights * outputOne;
        outputTwo = netTwo[1];

        absoluteError = outputTwo - expectedOutput[i];
        global cost += absoluteError .^ 2;
        percentError = abs(absoluteError / expectedOutput[i]) * 100;

        print("Percent error at test point ", i, ": ", percentError, "% \n")
end

global cost /= numOfTestPoints;

print("Total MSE: ", cost, '\n')
