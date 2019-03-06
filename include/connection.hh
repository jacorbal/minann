/**
 * @file connection.hh
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

#ifndef CONNECTION_HH
#define CONNECTION_HH

#include <cstdlib> // Randomizing related functions


namespace MinAnn {

/**
 */
class Connection {
  public:
    // LIFE CYCLE
    /**
     *
     * @note In this particular case, due the simplicity of this
     *       constructor, itgoes inline
     */
    Connection(void) {
        weight_ = rand() / double(RAND_MAX);
    }


    // ACCESSORS AND MUTATORS
    /**
     */
    void Weight(const double weight);

    /**
     */
    double Weight(void) const;

    /**
     */
    void DeltaWeight(const double delta_weight);

    /**
     */
    double DeltaWeight(void) const;


  private:
    double weight_;
    double delta_weight_;
};


// INLINE METHODS
inline void
Connection::Weight(const double weight) {
    weight_ = weight;
}


inline double
Connection::Weight(void) const
{
    return weight_;
}


inline void
Connection::DeltaWeight(const double delta_weight) {
    delta_weight_ = delta_weight;
}


inline double
Connection::DeltaWeight(void) const
{
    return delta_weight_;
}


} // ! namespace MinAnn


#endif // ! CONNECTION_HH

