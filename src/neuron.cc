/**
 * @file neuron.hh
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

#include <cmath>
#include <neuron.hh>


namespace MinAnn {

// CONSTANTS
double Neuron::kEta = 0.15f;    // Overall net learning rate
double Neuron::kAlpha = 0.5f;   // Momentum; multiplier of last delta_weight


// PUBLIC =============================================================

// LIFE CYCLE ---------------------------------------------------------
Neuron::Neuron(unsigned num_outputs, unsigned index)
{
    for (unsigned c = 0; c < num_outputs; ++c) {
        output_weights_.push_back(Connection());
    }

    index_ = index;
}


Neuron::~Neuron(void)
{
    output_weights_.clear();
}


// OPERATIONS ---------------------------------------------------------
void
Neuron::FeedForward(const std::vector<Neuron>& prev_layer)
{
    double sum = 0.0f;

    /* Sum the previous layer's outputs (now our inputs); and include
     * the bias node from the previous layer */
    for (unsigned n = 0; n < prev_layer.size(); ++n) {
        sum += prev_layer[n].OutputValue() *
            prev_layer[n].output_weights_[index_].Weight();
    }

    output_value_ = Neuron::TransferFunction(sum);
}



void
Neuron::UpdateInputWeights(std::vector<Neuron>& prev_layer)
{
    /* The weights to be updated are in the `Connection' container in
     * the neurons in the preceding layer */
    for (unsigned n = 0; n < prev_layer.size(); ++n) {
        Neuron& neuron = prev_layer[n];

        // Remember other neurons connection weights from it to "us"
        double old_delta_weight =
            neuron.output_weights_[index_].DeltaWeight();
        // Individual input, magnified by the gradient and the train rate.
        double new_delta_weight = kEta *        // overall learning rate
                                  neuron.OutputValue() *
                                  gradient_ +
                                  kAlpha *      // momentum
                                  old_delta_weight;

        neuron.output_weights_[index_].DeltaWeight(new_delta_weight);
        neuron.output_weights_[index_].Weight(new_delta_weight +
                neuron.output_weights_[index_].Weight());
    }
}


void
Neuron::CalcHiddenGradients(const std::vector<Neuron>& next_layer)
{
    double dow = SumDow(next_layer);
    gradient_ = dow * Neuron::TransferFunctionDerivative(output_value_);
}


void
Neuron::CalcOutputGradients(double target_value)
{
    double delta = target_value - output_value_;
    gradient_ = delta * Neuron::TransferFunctionDerivative(output_value_);
}


double
Neuron::TransferFunction(double x)
{
    // σ(x) = (1 + exp(-x))^(-1) [=== sigmoid] in [0., 1.]
    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) in [-1., 1.]
    return tanh(x);
}


double
Neuron::TransferFunctionDerivative(double x)
{
    // d/dx(σ(x)) = exp(x) / (1 + exp(x))^2
    // d/dx(tanh(x)) = 1 - (tanh(x))^2 ~= 1 - x^2
    return 1.0f - x * x;
}


// PRIVATE ============================================================

// OPERATIONS ---------------------------------------------------------
double
Neuron::SumDow(const std::vector<Neuron>& next_layer) const
{
    double sum = 0.0f;

    // Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < next_layer.size() - 1; ++n) {
        sum += output_weights_[n].Weight() * next_layer[n].gradient_;
    }

    return sum;
}


} // ! namespace MinAnn

