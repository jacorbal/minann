/**
 * @file net.cc
 */
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

#include <cassert>
#include <cmath>
#include <vector>

#include <neuron.hh>
#include <net.hh>

namespace MinAnn {

// CONSTANTS
double Net::recent_avg_smoothing_factor_ = 100.0f;


// PUBLIC =============================================================

// LIFE CYCLE ---------------------------------------------------------
Net::Net(const std::vector<unsigned>& topology)
{
    unsigned numLayers = topology.size();

    for (unsigned layer_num = 0; layer_num < numLayers; ++layer_num) {
        layers_.push_back(Layer());
        unsigned num_outputs = layer_num == topology.size() - 1
            ? 0
            : topology[layer_num + 1];

        /* We have a new layer, now fill it with neurons, and add a bias
         * neuron in each layer */
        for (unsigned neuron_num = 0;
             neuron_num <= topology[layer_num];
             ++neuron_num) {
            layers_.back().push_back(Neuron(num_outputs, neuron_num));
        }

        /* Force the bias node's output to 1.0 (it was the last neuron
         * pushed in this layer) */
        layers_.back().back().OutputValue(1.0f);
    }
}


Net::~Net(void)
{
    layers_.clear();
}


// OPERATIONS ---------------------------------------------------------
void
Net::FeedForward(const std::vector<double>& input_values)
{
    assert(input_values.size() == layers_[0].size() - 1);

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < input_values.size(); ++i) {
        layers_[0][i].OutputValue(input_values[i]);
    }

    // Forward propagate
    for (unsigned layer_num = 1;
         layer_num < layers_.size();
         ++layer_num) {
        Layer& prev_layer = layers_[layer_num - 1];
        for (unsigned n = 0; n < layers_[layer_num].size() - 1; ++n) {
            layers_[layer_num][n].FeedForward(prev_layer);
        }
    }
}


void
Net::BackPropagation(const std::vector<double>& target_values)
{
    // Calculate overall net error (RMS of output neuron errors)
    Layer& output_layer = layers_.back();
    error_ = 0.0f;

    for (unsigned n = 0; n < output_layer.size() - 1; ++n) {
        double delta = target_values[n] - output_layer[n].OutputValue();
        error_ += delta * delta;
    }
    error_ /= output_layer.size() - 1; // Get avg. error squared
    error_ = sqrt(error_);             // RMS (root mean square error)

    // Implement a recent average measurement
    recent_avg_error_ =
        (recent_avg_error_ * recent_avg_smoothing_factor_ + error_) /
            (recent_avg_smoothing_factor_ + 1.0f);

    // Calculate output layer gradients
    for (unsigned n = 0; n < output_layer.size() - 1; ++n) {
        output_layer[n].CalcOutputGradients(target_values[n]);
    }

    // Calculate hidden layer gradients
    for (unsigned layer_num = layers_.size() - 2;
         layer_num > 0;
         --layer_num) {
        Layer& hidden_layer = layers_[layer_num];
        Layer& next_layer = layers_[layer_num + 1];

        for (unsigned n = 0; n < hidden_layer.size(); ++n) {
            hidden_layer[n].CalcHiddenGradients(next_layer);
        }
    }

    /* For all layers from outputs to first hidden layer, update
     * connection weights */
    for (unsigned layer_num = layers_.size() - 1;
         layer_num > 0;
         --layer_num) {
        Layer& layer = layers_[layer_num];
        Layer& prev_layer = layers_[layer_num - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].UpdateInputWeights(prev_layer);
        }
    }
}


void
Net::Results(std::vector<double>& result_values) const
{
    result_values.clear();

    for (unsigned n = 0; n < layers_.back().size() - 1; ++n) {
        result_values.push_back(layers_.back()[n].OutputValue());
    }
}


} // ! namespace MinAnn

