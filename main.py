import csv
from numpy import exp, array, random, dot


def conv(s):
    try:
        s = int(s)
    except ValueError:
        pass
    return s

# Read from csv file and fill list of inputs and ouputs
# inputs are the first 5 columns and the last columns is the output
inputs_l = []
outputs_l = []
with open('data.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        # Inputs list
        inputs_l.append([conv(x) for x in row[:5]])
        # Output list
        outputs_l.append(conv(row[5]))

# Input and Output convert list to numpy arrays
inputs = array(inputs_l)
outputs = array([outputs_l]).T     # .T is to transpose, y matrix has 5 rows with 1 column (vertical form)

# unknown input
new_input = array([1,1,1,0,1])

# initialize synapse_weights
random.seed(1)
# make use of machine learning formula Delta rule.
weights = 2 * random.random((5,1)) - 1

# Test here to see possible weights values
# print weights

# output without untrained neuron usign Sigmoid Function
# which always return a value between -1 or 1: f(x) = 1 / 1 + e exp (-(input * weight))
# print 1 / (1 + exp(-(dot(new_input, weights))))

# Train the network in a range of 10000 iterations
for i in xrange(10000):
    # Calculate the value for the each of the examples
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    # Run the adjustments to weights
    weights += dot(inputs.T, (outputs - output) * (output * (1 - output)))
    # print "Neural Network is learning ..."

# Test here to see Weights after training
# print weights

value_with_train = 1 / (1 + exp(-(dot(new_input, weights))))
if value_with_train > 0.5:
    result = "### The result could be: 1 based on sigmoid function and trainings: {0}".format(value_with_train)
elif value_with_train < 0.5:
    result = "### The result could be: 0 on sigmoid function and trainings: {0}".format(value_with_train)
else:
    result = "### Try result the neural with more iterations, because it does not determine the result"

# Print the result for our unknown input
print "#"*40
print "#"*40
print result
print "#"*40
print "#"*40
