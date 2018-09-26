#ifndef _CAN_OPERATE_WEIGHTS_ERROR_FUNCTION_H_
#define _CAN_OPERATE_WEIGHTS_ERROR_FUNCTION_H_

#include "errorfunction.h"

class CanOperateWeightsErrorFunction : public ErrorFunction
{
public:
	virtual ~CanOperateWeightsErrorFunction();

	virtual ErrorFunctionType get_type() const;

	virtual Vector<Matrix<Data> > &assign_global_weight(Vector<Vector<Matrix<Data> > > &weight_vector_vector_matrix, Vector<Matrix<Data> > &ret) const = 0;
	virtual Vector<Matrix<Data> > &assign_global_weight_multi_thread(Vector<Vector<Matrix<Data> > > &weight_vector_vector_matrix, Vector<Matrix<Data> > &ret) const = 0;

	virtual MWVector<Matrix<Data> > get_global_weight_structure(const MWVector<MWVector<Matrix<Data> > > &weight_vector_vector_matrix) const = 0;

	MWVector<Matrix<Data> > get_global_weight(MWVector<MWVector<Matrix<Data> > > &weight_vector_vector_matrix) const;
};

#endif

