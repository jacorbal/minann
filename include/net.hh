/**
 * @file net.hh
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

#ifndef NET_HH
#define NET_HH

#include <vector>


namespace MinAnn {

class Neuron;


/**
 */
class Net {
  public:
    // LIFE CYCLE
    /**
     */
    Net(const std::vector<unsigned>& topology);

    /**
     */
    ~Net(void);


    // OPERATIONS
    /**
     */
    void FeedForward(const std::vector<double>& input_values);

    /**
     *
     * @note It uses root mean square error (@e rms), where
     *   @f\$rms=\sqrt{\frac{1}{n}\sum_i{\left(target_i-actual_i\right)^2}}@f\$
     */
    void BackPropagation(const std::vector<double>& target_values);

    /**
     */
    void Results(std::vector<double>& result_values) const;


    // ACCESSORS AND MUTATORS
    /**
     */
    double RecentAvgError(void) const;


  private:
    typedef std::vector<Neuron> Layer;

    std::vector<Layer> layers_; ///< ?
    double error_;              ///< ?
    double recent_avg_error_;   ///< ?
    static double recent_avg_smoothing_factor_; /**< Number of training
                                                     samples to avg. over */
};


// INLINE METHODS
inline double
Net::RecentAvgError(void) const
{
    return recent_avg_error_;
}


} // ! namespace MinAnn


#endif // ! NET_HH

