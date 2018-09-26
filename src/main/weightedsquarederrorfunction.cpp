#include <vectormatrixmultithreadcalculator.h>
#include <mwmathfunction.h>
#include <mwmatrixhelper.h>

#include "weightedsquarederrorfunction.h"

WeightedSquaredErrorFunction::WeightedSquaredErrorFunction(const Vector<Data> &error_weight_in_unit_vector, const MWMathFunction* error_weight_function)
	: _error_weight_function(error_weight_function)
	, _error_weight_in_unit_matrix(Matrix<Data>(error_weight_in_unit_vector, MWVectorAsMatrixTypeRow))
{
}

Data WeightedSquaredErrorFunction::get_error(const Vector<Matrix<Data> > &a_vector_matrix, const Vector<Matrix<Data> > &t_vector_matrix, Vector<Matrix<Data> > &e_vector_matrix, Vector<Matrix<Data> > &e_vector_matrix_mid) const
{
	LOG_TRACE(_logger, "a_vector_matrix = " << a_vector_matrix);
	VectorMatrixMultiThreadCalculator::assign_minus(t_vector_matrix, a_vector_matrix, e_vector_matrix);
	LOG_TRACE(_logger, "e_vector_matrix = " << e_vector_matrix);

	Matrix<Data> error(e_vector_matrix[0].get_matrix_info(), 0);
	VectorMatrixMultiThreadCalculator::assign_square(e_vector_matrix, e_vector_matrix_mid);
	// VectorMatrixMultiThreadCalculator::assign_sum(e_vector_matrix_mid, error);
	assign_sum(e_vector_matrix_mid, error);

	return error._data[0][0] / e_vector_matrix.size();
}

Data WeightedSquaredErrorFunction::get_error(const Vector<Matrix<Data> > &a_vector_matrix, const Vector<Matrix<Data> > &t_vector_matrix) const
{
	Vector<Matrix<Data> > e_vector_matrix = a_vector_matrix, e_vector_matrix_mid = a_vector_matrix;
	return get_error(a_vector_matrix, t_vector_matrix, e_vector_matrix, e_vector_matrix_mid);
}

Matrix<Data> WeightedSquaredErrorFunction::get_error_wegiht_in_unit_vector(const Vector<Data> &error_weight_in_unit_vector) const
{
	return _error_weight_in_unit_matrix;
}

void WeightedSquaredErrorFunction::set_error_wegiht_in_unit_vector(const Vector<Data> &error_weight_in_unit_vector)
{
	_error_weight_in_unit_matrix = Matrix<Data>(error_weight_in_unit_vector, MWVectorAsMatrixTypeRow);
}

void WeightedSquaredErrorFunction::set_error_weight_function(const MWMathFunction* error_weight_function)
{
	_error_weight_function = error_weight_function;
}

const MWMathFunction* WeightedSquaredErrorFunction::get_error_weight_function() const
{
	return _error_weight_function;
}

MWVector<Matrix<Data> > WeightedSquaredErrorFunction::get_global_weight_structure(const MWVector<MWVector<Matrix<Data> > > &weight_vector_vector_matrix) const
{
	MWVector<Matrix<Data> > ret;

	size_t i;
	for (i = 0; i < weight_vector_vector_matrix.size(); ++ i)
	{
		if (weight_vector_vector_matrix[i].empty())
		{
			LOG_ERROR(_logger, "weight_vector_vector_matrix[" << i <<"] is empty. return.");
			return ret;
		}
		ret.push_back(Matrix<Data>(weight_vector_vector_matrix[i][0].get_matrix_info()));
	}
	if (i == 0)
	{
		LOG_WARN(_logger, "WeightedSquaredErrorFunction::get_global_weight(an_empty_weight_vector_vector). return.");
	}

	return ret;
}

MWVector<Matrix<Data> > &WeightedSquaredErrorFunction::assign_global_weight(MWVector<MWVector<Matrix<Data> > > &weight_vector_vector_matrix, MWVector<Matrix<Data> > &ret) const
{
	const size_t vector_size(weight_vector_vector_matrix.size() == 0 ? 0 : weight_vector_vector_matrix[0].size());
	if (vector_size == 0)
	{
		LOG_ERROR(_logger, "weight_vector_vector_matrix.size() == 0 or weight_vector_vector_matrix[0].size() == 0");
		return ret;
	}

	for (size_t i = 0, i_layer; i < weight_vector_vector_matrix[0].size(); ++ i)
	{
		for (i_layer = 0; i_layer < weight_vector_vector_matrix.size(); ++ i_layer)
		{
			if (i == 0)
			{
				weight_vector_vector_matrix[i_layer][i].assign(ret[i_layer]);
			}
			else
			{
				assign_plus(ret[i_layer], weight_vector_vector_matrix[i_layer][i], ret[i_layer]);
			}
		}
	}

	assign_times(ret, 1 / ((MWData) vector_size), ret);

	return ret;
}

MWVector<Matrix<Data> > &WeightedSquaredErrorFunction::assign_global_weight_multi_thread(MWVector<MWVector<Matrix<Data> > > &weight_vector_vector_matrix, MWVector<Matrix<Data> > &ret) const
{
	const size_t vector_size(weight_vector_vector_matrix.size() == 0 ? 0 : weight_vector_vector_matrix[0].size());
	if (vector_size == 0)
	{
		LOG_ERROR(_logger, "weight_vector_vector_matrix.size() == 0 or weight_vector_vector_matrix[0].size() == 0");
		return ret;
	}

	for (size_t i_layer = 0; i_layer < weight_vector_vector_matrix.size(); ++ i_layer)
	{
		// VectorMatrixMultiThreadCalculator::assign_sum(weight_vector_vector_matrix[i_layer], ret[i_layer]);
		assign_sum(weight_vector_vector_matrix[i_layer], ret[i_layer]);
	}

	assign_times(ret, 1 / ((MWData) vector_size), ret);

	return ret;
}
