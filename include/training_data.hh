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

#ifndef TRAINING_DATA_HH
#define TRAINING_DATA_HH

#include <vector>
#include <fstream>
#include <sstream>


/* Read training data from a file
 */
class TrainingData
{
  public:
    /**
     */
    TrainingData(const std::string filename)
    {
        training_data_file_.open(filename.c_str());
    }

    /**
     */
    bool IsEof(void)
    {
        return training_data_file_.eof();
    }

    /**
     */
    void Topology(std::vector<unsigned>& topology)
    {
        std::string line;
        std::string label;

        getline(training_data_file_, line);
        std::stringstream ss(line);
        ss >> label;
        if (this->IsEof() || label.compare("Topology:") != 0 ) {
            abort();
        }

        while (!ss.eof()) {
            unsigned n;
            ss >> n;
            topology.push_back(n);
        }
    }

    /**
     * @brief Returns the number of input values read from a a file
     */
    unsigned NextInputs(std::vector<double>& input_values)
    {
        input_values.clear();

        std::string line;
        getline(training_data_file_, line);
        std::stringstream ss(line);

        std::string label;
        ss >> label;
        if (label.compare("i:") == 0) {
            double value;
            while (ss >> value) {
                input_values.push_back(value);
            }
        }

        return input_values.size();
    }

    /**
     */
    unsigned TargetOutputs(std::vector<double>& target_output_values)
    {
        target_output_values.clear();

        std::string line;
        getline(training_data_file_, line);
        std::stringstream ss(line);

        std::string label;
        ss >> label;
        if (label.compare("o:") == 0) {
            double value;
            while (ss >> value) {
                target_output_values.push_back(value);
            }
        }

        return target_output_values.size();
    }


  private:
    std::ifstream training_data_file_;
};


#endif // ! TRAINING_DATA_HH

