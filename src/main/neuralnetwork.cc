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


#include <logsigfunction.h>
#include <linearfunction.h>
#include <randomfunction.h>
#include <mwmatrixhelper.h>
#include <vectormatrixmultithreadcalculator.h>

#include "neuralnetwork.h"

void NeuralNetwork::AddSequence() {
    AddSequenceData(&_neuron_num_vector);
    AddSequenceData(&_w_vector_matrix);
    AddSequenceData(&_b_vector_matrix);
}

NeuralNetwork::NeuralNetwork(size_t input_num, size_t hidden_num, size_t output_num)
    : _neuron_num_vector(Vector<size_t>(NETWORK_LAYER_NUM + 1))
    , _active_function_vector(std::vector<MWDerivableMathFunction*>(NETWORK_LAYER_NUM))
    , _w_vector_matrix(Vector<Matrix<Data> >(NETWORK_LAYER_NUM))
    , _b_vector_matrix(Vector<Matrix<Data> >(NETWORK_LAYER_NUM)) {
    Init(input_num, hidden_num, output_num);
}

void NeuralNetwork::Init(size_t input_num, size_t hidden_num, size_t output_num) {
    _neuron_num_vector[0] = input_num;
    _neuron_num_vector[1] = hidden_num;
    _neuron_num_vector[2] = output_num;

    Vector<Data> w_b_random_min_max(2);
    w_b_random_min_max[0] = W_B_RANDOM_MIN;
    w_b_random_min_max[1] = W_B_RANDOM_MAX;

    for (size_t i_layer = 0, i, j; i_layer < NETWORK_LAYER_NUM; ++ i_layer) {
        _w_vector_matrix[i_layer] = Matrix<Data>(_neuron_num_vector[i_layer + 1],
                                                 _neuron_num_vector[i_layer]);
        _b_vector_matrix[i_layer] = Matrix<Data>(_neuron_num_vector[i_layer + 1], 1);

        // 赋随机值
        for (i = 0; i < _neuron_num_vector[i_layer + 1]; ++ i) {
#ifdef _COMPILE_MODE_TEST_
            _b_vector_matrix[i_layer]._data[i][0] = 0;
#else
            _b_vector_matrix[i_layer]._data[i][0] =
                RandomFunction::GetInstance()->GetValue(w_b_random_min_max)[0];
#endif
            for (j = 0; j < _neuron_num_vector[i_layer]; ++ j) {
#ifdef _COMPILE_MODE_TEST_
                _w_vector_matrix[i_layer]._data[i][j] = 0;
#else
                _w_vector_matrix[i_layer]._data[i][j] =
                    RandomFunction::GetInstance()->GetValue(w_b_random_min_max)[0];
#endif
            }
        }
    }

    _active_function_vector[0] = LogsigFunction::GetInstance();
    _active_function_vector[1] = LinearFunction::GetInstance();
}

void NeuralNetwork::SetActiveFunction(MWDerivableMathFunction *active_function_arr[]) {
    for (size_t i = 0; i < NETWORK_LAYER_NUM; ++ i) {
        _active_function_vector[i] = active_function_arr[i];
    }
}

void NeuralNetwork::InitializeActiveFunction() {
    MWDerivableMathFunction *functions[2];
    functions[0] = LogsigFunction::GetInstance();
    functions[1] = LinearFunction::GetInstance();
    SetActiveFunction(functions);
}

const MWDerivableMathFunction *NeuralNetwork::GetActiveFunction(
    size_t i_layer) {
    if (i_layer >= NETWORK_LAYER_NUM) {
        LOGERROR("(i_layer = " << i_layer << 
                 ") >= (NETWORK_LAYER_NUM = " << NETWORK_LAYER_NUM << 
                 "), can not GetActiveFunction, return NULL");
        return NULL;
    }

    return _active_function_vector[i_layer];
}

void NeuralNetwork::AssignOutputStructure(size_t vector_size,
                                          Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix,
                                          Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix,
                                          Vector<Matrix<Data> > &a_end_vector_matrix) const {
    n_vector_vector_matrix = Vector<Vector<Matrix<Data> > >(NETWORK_LAYER_NUM);
    a_vector_vector_matrix = Vector<Vector<Matrix<Data> > >(NETWORK_LAYER_NUM);
    for (size_t i_layer = 0; i_layer < NETWORK_LAYER_NUM; ++ i_layer) {
        n_vector_vector_matrix[i_layer] = 
            Vector<Matrix<Data> >(vector_size, Matrix<Data>(_neuron_num_vector[i_layer + 1], 1));
        (i_layer + 1 < NETWORK_LAYER_NUM ? a_vector_vector_matrix[i_layer + 1] : 
                                           a_end_vector_matrix) = n_vector_vector_matrix[i_layer];
    }

    return;
}

Vector<Matrix<Data> > &NeuralNetwork::AssignOutput(
    Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix,
    Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix,
    Vector<Matrix<Data> > &a_end_vector_matrix,
    const Vector<Matrix<Data> > &w_vector_matrix,
    const Vector<Matrix<Data> > &b_vector_matrix) const {
    for (size_t i_layer = 0; i_layer < NETWORK_LAYER_NUM; ++ i_layer) {
        _active_function_vector[i_layer]->AssignValueVectorMatrixMultiThread(
            VectorMatrixMultiThreadCalculator::AssignPlus(
                VectorMatrixMultiThreadCalculator::AssignTimes(w_vector_matrix[i_layer],
                                                               a_vector_vector_matrix[i_layer], 
                                                               n_vector_vector_matrix[i_layer]),
                b_vector_matrix[i_layer], n_vector_vector_matrix[i_layer]),
            (i_layer + 1 == NETWORK_LAYER_NUM) ? a_end_vector_matrix :
            a_vector_vector_matrix[i_layer + 1]);
    }

    return a_end_vector_matrix;
}

Vector<Matrix<Data> > &NeuralNetwork::AssignOutput(
    Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix,
    Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix,
    Vector<Matrix<Data> > &a_end_vector_matrix) const {
    return AssignOutput(n_vector_vector_matrix, 
                        a_vector_vector_matrix,
                        a_end_vector_matrix, 
                        _w_vector_matrix, 
                        _b_vector_matrix);
}

Vector<Matrix<Data> > NeuralNetwork::GetOutput(const Vector<Matrix<Data> > &p)
    const {
    Vector<Matrix<Data> > a_end_vector_matrix;
    Vector<Vector<Matrix<Data> > > n_vector_vector_matrix, 
                                   a_vector_vector_matrix;
    AssignOutputStructure(p.size(), 
                          n_vector_vector_matrix,
                          a_vector_vector_matrix, 
                          a_end_vector_matrix);

    a_vector_vector_matrix[0] = p;
    AssignOutput(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix);
    return a_end_vector_matrix;
}


