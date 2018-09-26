#ifndef _ERROR_FUNCTION_H_
#define _ERROR_FUNCTION_H_

#include <mwmatrix.h>

#include "neuralnetworkglobal.h"

enum ErrorFunctionType
{
	ErrorFunctionTypeCanOperateWeights,
	ErrorFunctionTypeCanNotOperateWeights
};

class ErrorFunction
{
public:
	virtual ~ErrorFunction();

	virtual Data get_error(const Vector<Matrix<Data> > &a_vector_matrix, const Vector<Matrix<Data> > &t_vector_matrix) const;

	virtual Data get_error(const Vector<Matrix<Data> > &a_vector_matrix, const Vector<Matrix<Data> > &t_vector_matrix, Vector<Matrix<Data> > &e_vector_matrix, Vector<Matrix<Data> > &e_vector_matrix_mid) const = 0;
	virtual ErrorFunctionType get_type() const = 0;
};

#endif
