// Micro Wave Neural Network
// Copyright (c) 2015-2020, Ark Lee
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// You must obey the GNU General Public License in all respects for
// all of the code used.  If you modify file(s) with this exception, 
// you may extend this exception to your version of the file(s), but 
// you are not obligated to do so.  If you do not wish to do so, delete 
// this exception statement from your version. If you delete this exception 
// statement from all source files in the program, then also delete it here.
//
// Contact:  Ark Lee <arklee@houduan.online>
//           Beijing, China


#include <iostream>
#include <float.h>

#include <mwglobal.h>
#include <mwlogger.h>

#include <mwstring.h>
#include <binfile.h>
#include <mwtimer.h>

#include <mwmathglobal.h>
#include <mwmatrix.h>
#include <vectormatrixmultithreadcalculator.h>

#include "neuralnetworkglobal.h"
#include "neuralnetwork.h"
#include "traininginformation.h"
#include "backpropgation.h"
#include "weightedsquarederrorfunction.h"


#define PROGRAMM_NAME "testneuralnetwork"

int main(int argc, const char *argv[]) {
    if (argc < 7) {
        std::cout << "usage: " << argv[0] <<
             " thread_num data_num hidden_neuron_num min_error network_file_name time_limit" << std::endl;
        return EXIT_FAILURE;
    }

    InitLogger(PROGRAMM_NAME);

    MWString network_file_name(argv[5]);
    BinFile network_file(network_file_name);
    if (!network_file.OpenForWrite()) {
        LOGERROR("open file " << network_file_name << " for write error");
        return EXIT_FAILURE;
    }

    const size_t data_num(MWString(argv[2]).ToSizeT());
    const size_t hidden_neuron_num(MWString(argv[3]).ToSizeT());

    LOGINFO("start, data_num = " << data_num);

    Vector<Matrix<Data> > p_vector_matrix(data_num * data_num, Matrix<Data>(2, 1)),
                          t_vector_matrix(data_num * data_num, Matrix<Data>(1, 1)),
                          test_p_vector_matrix(data_num * data_num, Matrix<Data>(2, 1)),
                          test_t_vector_matrix(data_num * data_num, Matrix<Data>(1, 1)),
                          a_minus_t_vector_matrix = test_t_vector_matrix;

    for (size_t i(0); i < data_num; ++i) {
        for (size_t j(0); j < data_num; ++j) {
            t_vector_matrix[i * data_num + j]._data[0][0] = 
                (p_vector_matrix[i * data_num + j]._data[0][0] = ((Data) (i + 1) / (data_num + 1) + 1)) *
                (p_vector_matrix[i * data_num + j]._data[1][0] = ((Data) (j + 1) / (data_num + 1) + 1));
            test_t_vector_matrix[i * data_num + j]._data[0][0] = 
                (test_p_vector_matrix[i * data_num + j]._data[0][0] = ((Data) i / data_num + 1)) *
                (test_p_vector_matrix[i * data_num + j]._data[1][0] = ((Data) j / data_num + 1));
        }
    }

    LOGTRACE("t_vector_matrix = " << t_vector_matrix);
    LOGTRACE("p_vector_matrix = " << p_vector_matrix);
    LOGTRACE("test_t_vector_matrix = " << test_t_vector_matrix <<
             ", test_p_vector_matrix = " << test_p_vector_matrix);

    const size_t thread_num = MWString(argv[1]).ToSizeT();
    VectorMatrixMultiThreadCalculator::Initialize(thread_num);
    MWMathFunction::Initialize(thread_num);

    NeuralNetwork network(2, hidden_neuron_num, 1);
    double min_error(MWString(argv[4]).ToDouble());
    double time_limit(MWString(argv[6]).ToDouble());
    double test_error;
    TrainingInformation stop_condition(min_error, 0, time_limit);
    TrainingInformation training_information;
    LOGINFO("training starts, network.get_neuron_num_vector() = " << network.GetNeuronNumVector() << 
            ", using min_error = " << min_error);
    size_t epoch(0);
    Vector<Vector<Matrix<Data> > > n_vector_vector_matrix,
                                   a_vector_vector_matrix;
    Vector<Matrix<Data> > a_end_vector_matrix;
    network.AssignOutputStructure(p_vector_matrix.size(), 
                                  n_vector_vector_matrix,
                                  a_vector_vector_matrix, 
                                  a_end_vector_matrix);
    a_vector_vector_matrix[0] = test_p_vector_matrix;
    while (true) {
        training_information = BackPropgation::GetInstance()->Train(&network,
                                                                     p_vector_matrix, 
                                                                     t_vector_matrix, 
                                                                     WeightedSquaredErrorFunction::GetInstance(),
                                                                     &stop_condition);
        test_error = WeightedSquaredErrorFunction::GetInstance()->GetError(network.AssignOutput(n_vector_vector_matrix, 
                                                                                                a_vector_vector_matrix,
                                                                                                a_end_vector_matrix), 
                                                                           test_t_vector_matrix);
        std::cout << "error = " << training_information.GetError() << 
                ", test_error = " << test_error << 
                ", epoch = " << (epoch += training_information.GetEpoch()) << std::endl;
        LOGINFO("error = " << training_information.GetError() <<
                ", test_error = " << test_error << 
                ", epoch = " << epoch);
        if (training_information.IsSatisfied(&stop_condition)) {
            break;
        }
        stop_condition.SetTimeLimit(stop_condition.GetTimeLimit() - training_information.GetCostTime());
        stop_condition.SetEpoch(stop_condition.GetEpoch() - training_information.GetEpoch());
    }
    LOGINFO("training finished");

    MWMathFunction::Finalize();
    VectorMatrixMultiThreadCalculator::Finalize();

    LOGINFO("network.get_w_vector_matrix() = " << network.GetWVectorMatrix() << 
            ", network.get_b_vector_matrix() = " << network.GetBVectorMatrix() << 
            ", network.get_neuron_num_vector() = " << network.GetNeuronNumVector());

    std::vector<char> char_vector = network.Serialize();
    if (!network_file.Write(char_vector)) {
        LOGERROR("write char_vector to " << network_file_name << " error");
        return EXIT_FAILURE;
    }

    LOGINFO("writing finished");

    return EXIT_SUCCESS;
}
