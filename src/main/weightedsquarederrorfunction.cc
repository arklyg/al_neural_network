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


#include <vectormatrixmultithreadcalculator.h>
#include <mwmathfunction.h>
#include <mwmatrixhelper.h>

#include "weightedsquarederrorfunction.h"

WeightedSquaredErrorFunction::WeightedSquaredErrorFunction(
    const Vector<Data> &error_weight_in_unit_vector,
    const MWMathFunction *error_weight_function)
    : _error_weight_function(error_weight_function)
    , _error_weight_in_unit_matrix(Matrix<Data>(error_weight_in_unit_vector,
                                                MWVectorAsMatrixTypeRow)) {
}

Data WeightedSquaredErrorFunction::GetError(const Vector<Matrix<Data> > &a_vector_matrix, 
                                            const Vector<Matrix<Data> > &t_vector_matrix,
                                            Vector<Matrix<Data> > &e_vector_matrix,
                                            Vector<Matrix<Data> > &e_vector_matrix_mid) const {
    LOGTRACE("a_vector_matrix = " << a_vector_matrix);
    VectorMatrixMultiThreadCalculator::AssignMinus(t_vector_matrix,
                                                   a_vector_matrix, 
                                                   e_vector_matrix);
    LOGTRACE("e_vector_matrix = " << e_vector_matrix);

    Matrix<Data> error(e_vector_matrix[0].GetMatrixInfo(), 0);
    VectorMatrixMultiThreadCalculator::AssignSquare(e_vector_matrix,
                                                    e_vector_matrix_mid);
    AssignSum(e_vector_matrix_mid, error);

    return error._data[0][0] / e_vector_matrix.size();
}

Matrix<Data> WeightedSquaredErrorFunction::GetErrorWeightInUnitVector(
    const Vector<Data> &error_weight_in_unit_vector) const {
    return _error_weight_in_unit_matrix;
}

void WeightedSquaredErrorFunction::SetErrorWeightInUnitVector(
    const Vector<Data> &error_weight_in_unit_vector) {
    _error_weight_in_unit_matrix = Matrix<Data>(error_weight_in_unit_vector,
                                                MWVectorAsMatrixTypeRow);
}

void WeightedSquaredErrorFunction::SetErrorWeightFunction(
    const MWMathFunction *error_weight_function) {
    _error_weight_function = error_weight_function;
}

const MWMathFunction *WeightedSquaredErrorFunction::GetErrorWeightFunction() const {
    return _error_weight_function;
}

MWVector<Matrix<Data> >
WeightedSquaredErrorFunction::GetGlobalWeightStructure(
    const MWVector<MWVector<Matrix<Data> > > &WeightVectorVectorMatrix) const {
    MWVector<Matrix<Data> > ret;

    size_t i;
    for (i = 0; i < WeightVectorVectorMatrix.size(); ++ i) {
        if (WeightVectorVectorMatrix[i].empty()) {
            LOGERROR("WeightVectorVectorMatrix[" << i <<"] is empty. return.");
            return ret;
        }
        ret.push_back(Matrix<Data>(WeightVectorVectorMatrix[i][0].GetMatrixInfo()));
    }
    if (i == 0) {
        LOGWARN("WeightedSquaredErrorFunction::get_global_weight(an_empty_weight_vector_vector). return.");
    }

    return ret;
}

MWVector<Matrix<Data> > &WeightedSquaredErrorFunction::AssignGlobalWeight(
    MWVector<MWVector<Matrix<Data> > > &WeightVectorVectorMatrix,
    MWVector<Matrix<Data> > &ret) const {
    const size_t vector_size(WeightVectorVectorMatrix.size() == 0 ? 0 :
                             WeightVectorVectorMatrix[0].size());
    if (vector_size == 0) {
        LOGERROR("WeightVectorVectorMatrix.size() == 0 or WeightVectorVectorMatrix[0].size() == 0");
        return ret;
    }

    for (size_t i = 0, i_layer; i < WeightVectorVectorMatrix[0].size(); ++ i) {
        for (i_layer = 0; i_layer < WeightVectorVectorMatrix.size(); ++ i_layer) {
            if (i == 0) {
                WeightVectorVectorMatrix[i_layer][i].Assign(ret[i_layer]);
            } else {
                AssignPlus(ret[i_layer], 
                           WeightVectorVectorMatrix[i_layer][i],
                           ret[i_layer]);
            }
        }
    }

    AssignTimes(ret, 1 / ((MWData) vector_size), ret);

    return ret;
}

MWVector<Matrix<Data> > &WeightedSquaredErrorFunction::AssignGlobalWeightMultiThread(
    MWVector<MWVector<Matrix<Data> > > &WeightVectorVectorMatrix,
    MWVector<Matrix<Data> > &ret) const {
    const size_t vector_size(WeightVectorVectorMatrix.size() == 0 ? 0 :
                             WeightVectorVectorMatrix[0].size());
    if (vector_size == 0) {
        LOGERROR("WeightVectorVectorMatrix.size() == 0 or WeightVectorVectorMatrix[0].size() == 0");
        return ret;
    }

    for (size_t i_layer = 0; i_layer < WeightVectorVectorMatrix.size(); ++ i_layer) {
        AssignSum(WeightVectorVectorMatrix[i_layer], ret[i_layer]);
    }

    AssignTimes(ret, 1 / ((MWData) vector_size), ret);

    return ret;
}
