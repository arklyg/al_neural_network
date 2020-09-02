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


#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#define W_B_RANDOM_MIN -1
#define W_B_RANDOM_MAX 1

#define NETWORK_LAYER_NUM 2

#define DEFAULT_INPUT_NUM 1
#define DEFAULT_HIDDEN_NUM 2
#define DEFAULT_OUTPUT_NUM 1

#include <mwmatrix.h>

#include <mwserializable.h>

#include "neuralnetworkglobal.h"

class MWDerivableMathFunction;

class NeuralNetwork : public MWSerializable {
  private:
    Vector<size_t> _neuron_num_vector;
    std::vector<MWDerivableMathFunction*> _active_function_vector;
    Vector<Matrix<Data> > _w_vector_matrix;
    Vector<Matrix<Data> > _b_vector_matrix;

  protected:
    virtual void AddSequence();

  public:

    NeuralNetwork(size_t input_num = DEFAULT_INPUT_NUM,
                  size_t hidden_num = DEFAULT_HIDDEN_NUM, 
                  size_t output_num = DEFAULT_OUTPUT_NUM);
    void Init(size_t input_num = DEFAULT_INPUT_NUM,
              size_t hidden_num = DEFAULT_HIDDEN_NUM, 
              size_t output_num = DEFAULT_OUTPUT_NUM);

    void SetActiveFunction(MWDerivableMathFunction *active_function_arr[]);
    void InitializeActiveFunction();
    const MWDerivableMathFunction *GetActiveFunction(size_t layer_num);

    inline const Vector<size_t> &GetNeuronNumVector() const;
    inline const Vector<Matrix<Data> > &GetWVectorMatrix() const;
    inline const Vector<Matrix<Data> > &GetBVectorMatrix() const;

    inline Vector<Matrix<Data> > &AssignWVectorMatrix(Vector<Matrix<Data> >
                                                      &w_vector_matrix) const;
    inline Vector<Matrix<Data> > &AssignBVectorMatrix(Vector<Matrix<Data> >
                                                      &b_vector_matrix) const;

    void AssignOutputStructure(size_t vector_size,
                               Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix,
                               Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix,
                               Vector<Matrix<Data> > &a_end_vector_matrix) const;
    Vector<Matrix<Data> > &AssignOutput(Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix, 
                                        Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix,
                                        Vector<Matrix<Data> > &a_end_vector_matrix,
                                        const Vector<Matrix<Data> > &w_vector_matrix,
                                        const Vector<Matrix<Data> > &b_vector_matrix) const;
    Vector<Matrix<Data> > &AssignOutput(Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix,
                                        Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix,
                                        Vector<Matrix<Data> > &a_end_vector_matrix) const;
    Vector<Matrix<Data> > GetOutput(const Vector<Matrix<Data> > &p) const;

    inline void SetWVectorMatrix(const Vector<Matrix<Data> > &w);
    inline void SetBVectorMatrix(const Vector<Matrix<Data> > &b);

    inline size_t GetLayerNum() const;
};

inline const Vector<size_t> &NeuralNetwork::GetNeuronNumVector() const {
    return _neuron_num_vector;
}

inline const Vector<Matrix<Data> > &NeuralNetwork::GetWVectorMatrix() const {
    return _w_vector_matrix;
}

inline const Vector<Matrix<Data> > &NeuralNetwork::GetBVectorMatrix() const {
    return _b_vector_matrix;
}

inline Vector<Matrix<Data> > &NeuralNetwork::AssignWVectorMatrix(
    Vector<Matrix<Data> > &w_vector_matrix) const {
    return Assign(_w_vector_matrix, w_vector_matrix);
}

inline Vector<Matrix<Data> > &NeuralNetwork::AssignBVectorMatrix(
    Vector<Matrix<Data> > &b_vector_matrix) const {
    return Assign(_b_vector_matrix, b_vector_matrix);
}

inline void NeuralNetwork::SetWVectorMatrix(const Vector<Matrix<Data> >
                                            &w_vector_matrix) {
    _w_vector_matrix = w_vector_matrix;
}
inline void NeuralNetwork::SetBVectorMatrix(const Vector<Matrix<Data> >
                                            &b_vector_matrix) {
    _b_vector_matrix = b_vector_matrix;
}

inline size_t NeuralNetwork::GetLayerNum() const {
    return NETWORK_LAYER_NUM;
}

#endif
