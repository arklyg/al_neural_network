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

class WeightedSquaredErrorFunction : public CanOperateWeightsErrorFunction, public MWSingleton<WeightedSquaredErrorFunction>
{
private:
	const MWMathFunction* _error_weight_function;
	Matrix<Data> _error_weight_in_unit_matrix;

public:
	WeightedSquaredErrorFunction(const Vector<Data> &error_weight_in_unit_vector = Vector<Data>(1, 1), const MWMathFunction* error_weight_function = ArithmeticMeanFunction::get_instance());

	Matrix<Data> get_error_wegiht_in_unit_vector(const Vector<Data> &error_weight_in_unit_vector) const;
	void set_error_wegiht_in_unit_vector(const Vector<Data> &error_weight_in_unit_vector);
	void set_error_weight_function(const MWMathFunction* error_function);

	const MWMathFunction* get_error_weight_function() const;
	Vector<Data> get_error_weight_in_unit_vector() const;

	Data get_error(const Vector<Matrix<Data> > &a_vector_matrix, const Vector<Matrix<Data> > &t_vector_matrix) const;

	virtual Data get_error(const Vector<Matrix<Data> > &a_vector_matrix, const Vector<Matrix<Data> > &t_vector_matrix, Vector<Matrix<Data> > &e_vector_matrix, Vector<Matrix<Data> > &e_vector_matrix_mid) const;

	virtual MWVector<Matrix<Data> > get_global_weight_structure(const MWVector<MWVector<Matrix<Data> > > &weight_vector_vector_matrix) const;
	
	virtual Vector<Matrix<Data> > &assign_global_weight(Vector<Vector<Matrix<Data> > > &weight_vector_vector_matrix, Vector<Matrix<Data> > &ret) const;
	virtual Vector<Matrix<Data> > &assign_global_weight_multi_thread(Vector<Vector<Matrix<Data> > > &weight_vector_vector_matrix, Vector<Matrix<Data> > &ret) const;
};

#endif
