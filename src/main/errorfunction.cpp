#include "errorfunction.h"

ErrorFunction::~ErrorFunction()
{
}

Data ErrorFunction::get_error(const Vector<Matrix<Data> > &a_vector_matrix, const Vector<Matrix<Data> > &t_vector_matrix) const
{
	Vector<Matrix<Data> > e_vector_matrix = t_vector_matrix, e_vector_matrix_mid = t_vector_matrix;
	return get_error(a_vector_matrix, t_vector_matrix, e_vector_matrix, e_vector_matrix_mid);
}

