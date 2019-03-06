/*
 * Copyright (c) 2019, J. A. Corbal
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */ 
/*
 * 'minann' usage sample file:
 * An example of training data could be:
 *
 *      Topology: 2 4 2
 *      i: 1.00000 0.00000
 *      o: 1.00000 0.00000
 *      i: 1.00000 0.00000
 *      o: 1.00000 0.00000
 *      i: 0.00000 0.00000
 *      o: 0.00000 0.00000
 *      ...
 *
 * where the first line indicates the topology of the net in:
 * inputs, hidden layers, outputs
 *
 * The number of values in every input or output must match
 * the topology of the net.
 */

#include <cassert>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include <net.hh>
#include <training_data.hh>


// Print a label and vector values to screen
void VectorVals(std::string label, std::vector<double>& v,
                std::string end_line="\n");


// Main entry
int main(void)
{
    std::vector<double> input_values, target_values, result_values;
    std::vector<unsigned> topology;
    TrainingData training_data("training_data.dat");
    training_data.Topology(topology);
    MinAnn::Net net(topology);
    int training_pass = 0;

    while (!training_data.IsEof()) {
        // Get new input data and feed it forward
        if (training_data.NextInputs(input_values) != topology[0]) {
            break;
        }

        // Print iteration number
        ++training_pass;
        std::cout << std::endl << "Iter #" << training_pass << ":" <<
            std::endl;

        VectorVals(":  Inputs:", input_values);
        net.FeedForward(input_values);

        // Collect the net's actual results
        net.Results(result_values);
        VectorVals(": Outputs:", result_values);
        assert(result_values.size() == topology.back());

        // Train the net what the outputs should have been
        training_data.TargetOutputs(target_values);
        VectorVals(": Targets:", target_values);
        assert(target_values.size() == topology.back());

        net.BackPropagation(target_values);

        // Report how well training is working, averaged over recent
        // samples
        std::cout << "  Net recent avg. error: " << net.RecentAvgError() <<
            std::endl;
    }
    std::cout << std::endl << "Done training!" << std::endl;

    // Using the net after training. Sending input and getting results
    std::vector<double> input;

    std::cout << "\n-------------------------------------\n" << std::endl;

    input = {0, 0};
    net.FeedForward(input);
    net.Results(result_values);
    VectorVals("IN: ", input, "  ::  ");
    VectorVals("OUT: ", result_values);

    input = {0, 1};
    net.FeedForward(input);
    net.Results(result_values);
    VectorVals("IN: ", input, "  ::  ");
    VectorVals("OUT: ", result_values);

    input = {1, 0};
    net.FeedForward(input);
    net.Results(result_values);
    VectorVals("IN: ", input, "  ::  ");
    VectorVals("OUT: ", result_values);

    input = {1, 1};
    net.FeedForward(input);
    net.Results(result_values);
    VectorVals("IN: ", input, "  ::  ");
    VectorVals("OUT: ", result_values);

    std::cout << "\n-------------------------------------\n" << std::endl;

    return 0;
}


void
VectorVals(std::string label, std::vector<double>& v, std::string end_line)
{
    std::cout << label << " ";
    std::cout << "{";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << std::setprecision(3) << v[i];
        if (i != v.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "}";
    std::cout << end_line;
}

