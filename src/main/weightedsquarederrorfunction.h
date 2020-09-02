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


#ifndef _WEIGHTED_SQUARED_ERROR_FUNCTION_H_
#define _WEIGHTED_SQUARED_ERROR_FUNCTION_H_

#include <mwsingleton.h>
#include <mwmatrix.h>
#include <arithmeticmeanfunction.h>

#include "neuralnetworkglobal.h"
#include "canoperateweightserrorfunction.h"

#include "errorfunction.h"

class MWMathFunction;
class ArithmeticMeanFunction;

class WeightedSquaredErrorFunction : public CanOperateWeightsErrorFunction,
    public MWSingleton<WeightedSquaredErrorFunction> {
  private:
    const MWMathFunction *_error_weight_function;
    Matrix<Data> _error_weight_in_unit_matrix;

  public:
    WeightedSquaredErrorFunction(const Vector<Data> &error_weight_in_unit_vector =
                                     Vector<Data>(1, 1), 
                                 const MWMathFunction *error_weight_function =
                                     ArithmeticMeanFunction::GetInstance());

    Matrix<Data> GetErrorWeightInUnitVector(const Vector<Data>
                                                &error_weight_in_unit_vector) const;
    void SetErrorWeightInUnitVector(const Vector<Data>
                                        &error_weight_in_unit_vector);
    void SetErrorWeightFunction(const MWMathFunction *error_function);

    const MWMathFunction *GetErrorWeightFunction() const;
    Vector<Data> GetErrorWeightInUnitVector() const;

    inline Data GetError(const Vector<Matrix<Data> > &a_vector_matrix,
                         const Vector<Matrix<Data> > &t_vector_matrix) const;

    virtual Data GetError(const Vector<Matrix<Data> > &a_vector_matrix,
                          const Vector<Matrix<Data> > &t_vector_matrix,
                          Vector<Matrix<Data> > &e_vector_matrix,
                          Vector<Matrix<Data> > &e_vector_matrix_mid) const;

    virtual MWVector<Matrix<Data> > GetGlobalWeightStructure(
        const MWVector<MWVector<Matrix<Data> > > &WeightVectorVectorMatrix) const;

    virtual Vector<Matrix<Data> > &AssignGlobalWeight(
        Vector<Vector<Matrix<Data> > > &WeightVectorVectorMatrix,
        Vector<Matrix<Data> > &ret) const;
    virtual Vector<Matrix<Data> > &AssignGlobalWeightMultiThread(
        Vector<Vector<Matrix<Data> > > &WeightVectorVectorMatrix,
        Vector<Matrix<Data> > &ret) const;
};

inline Data WeightedSquaredErrorFunction::GetError(const Vector<Matrix<Data> > &a_vector_matrix, 
                                                   const Vector<Matrix<Data> > &t_vector_matrix) const {
    Vector<Matrix<Data> > e_vector_matrix = a_vector_matrix,
                          e_vector_matrix_mid = a_vector_matrix;
    return GetError(a_vector_matrix, t_vector_matrix, e_vector_matrix, e_vector_matrix_mid);
}

#endif
