# datagen.jl -- Generates CSV file containing 4 rows:
#               Row 1 contains resistor values from 0 to 100k Ohms
#               Row 2 contains capacitor values from 0 to 1u Farad
#               Row 3 contains times from 0 to 5RC (elementwise)
#               Row 4 contains outputs
#
# Accepts 2 inputs:
#               Input 1 is the number of datpoints to generate.
#               Input 2 is the filename to save the results to.

using Random
using DelimitedFiles

numRows = parse(UInt64, ARGS[1]);
vInitial = 10; # Value at t = 0

data = rand!(zeros(4,numRows));
data[1,:] *= 100000; # Put resistor column at 10k order of magnitude
data[2,:] /= 100000; # Put capacitor column at 1u order of magnitude

for i = 1:numRows
        data[3,i] *= 5*data[1,i]*data[2,i]; # Keeps t less than 5RC at each point
end

for i = 1:numRows
        data[4,i] = vInitial * exp(-data[3,i] / (data[1,i]*data[2,i])); # Ckt output value
end

writedlm(ARGS[2], data, ','); # Creates CSV file for data array.
