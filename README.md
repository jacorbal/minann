# Minimal artificial neural network

In machine learning and cognitive science, artificial neural networks
(ANNs) are a family of models inspired by biological neural networks
(the central nervous systems of animals, in particular the brain) and
are used to estimate or approximate functions that can depend on a large
number of inputs and are generally unknown. Artificial neural networks
are generally presented as systems of interconnected "neurons" which
exchange messages between each other. The connections have numeric
weights that can be tuned based on experience, making neural nets
adaptive to inputs and capable of learning.

Backpropagation, an abbreviation for "backward propagation of errors",
is a common method of training artificial neural networks used in
conjunction with an optimization method such as gradient descent. The
method calculates the gradient of a loss function with respect to all
the weights in the network. The gradient is fed to the optimization
method which in turn uses it to update the weights, in an attempt to
minimize the loss function.

The program takes the training data from a file called
`training_data.data`.

Usage sample file:

An example of training data could be:

     Topology: 2 4 2
     i: 1.00000 0.00000
     o: 1.00000 0.00000
     i: 1.00000 0.00000
     o: 1.00000 0.00000
     i: 0.00000 0.00000
     o: 0.00000 0.00000
     ...

where the first line indicates the topology of the net in: inputs,
hidden layers, outputs.  The number of values in every input or output
must match the topology of the net.

---

J. A. Corbal, 2019.

