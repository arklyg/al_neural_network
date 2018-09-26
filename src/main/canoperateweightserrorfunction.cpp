#include "canoperateweightserrorfunction.h"

CanOperateWeightsErrorFunction::~CanOperateWeightsErrorFunction()
{
}

ErrorFunctionType CanOperateWeightsErrorFunction::get_type() const
{
	return ErrorFunctionTypeCanOperateWeights;
}

MWVector<Matrix<Data> > CanOperateWeightsErrorFunction::get_global_weight(MWVector<MWVector<Matrix<Data> > > &weight_vector_vector_matrix) const
{
	MWVector<Matrix<Data> > ret = get_global_weight_structure(weight_vector_vector_matrix);
	return assign_global_weight(weight_vector_vector_matrix, ret);
}
