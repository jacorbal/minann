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

#ifndef NEURON_HH
#define NEURON_HH

#include <cstdlib> // Randomizing related functions
#include <vector>

#include <connection.hh>


namespace MinAnn {

/**
 */
class Neuron {
  public:
    // LIFE CYCLE
    /**
     */
    Neuron(unsigned num_outputs, unsigned index);

    /**
     */
    ~Neuron(void);


    // OPERATIONS
    /**
     *
     * @note output = @f$f(\sum_{i=0}^{n} x_i w_i)@f$
     */
    void FeedForward(const std::vector<Neuron>& prev_layer);

    /**
     */
    void UpdateInputWeights(std::vector<Neuron>& prev_layer);

    /**
     */
    void CalcOutputGradients(double target_value);

    /**
     */
    void CalcHiddenGradients(const std::vector<Neuron>& next_layer);


    // ACCESSORS AND MUTATORS
    /**
     */
    void OutputValue(const double value);

    /**
     */
    double OutputValue(void) const;


  private:
    /**
     * @brief Overall learning rate [0., 1.]
     *
     * @details Overall net training rate:
     *        - @e kEta = 0.0, means slow learner;
     *        - @e kEta = 0.2, medium learner;
     *        - @e kEta = 1.0, reckless learner;
     */
    static double kEta;

    /**
     * @brief Momentum [0., n]
     *
     * @details Multiplier of last weight change (of last delta weight):
     *        - @e kAlpha = 0.0, no momentum;
     *        - @e kAlpha = 0.5, moderate momentum;
     */
    static double kAlpha;

    double output_value_;
    std::vector<Connection> output_weights_;
    unsigned index_;
    double gradient_;

    /**
     * @brief Sum difference of weights
     */
    double SumDow(const std::vector<Neuron>& next_layer) const;

    /**
     */
    static double TransferFunction(double x);

    /**
     */
    static double TransferFunctionDerivative(double x);
};


// INLINE METHODS
inline void
Neuron::OutputValue(const double value) {
    output_value_ = value;
}


inline double
Neuron::OutputValue(void) const
{
    return output_value_;
}


} // ! namespace MinAnn


#endif // ! NEURON_HH

