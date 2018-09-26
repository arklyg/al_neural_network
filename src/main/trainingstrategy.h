#ifndef _TRAINING_STRATEGY_H_
#define _TRAINING_STRATEGY_H_

#include <mwmatrix.h>
#include <mwstring.h>

#include "traininginformation.h"

class NeuralNetwork;
class CanOperateWeightsErrorFunction;

class TrainingStrategy
{
public:
	virtual ~TrainingStrategy();

	virtual TrainingInformation train(NeuralNetwork* network, const MWVector<Matrix<Data> > &p_vector_matrix, const MWVector<Matrix<Data> > &t_vector_matrix, const CanOperateWeightsErrorFunction* error_function, const TrainingInformation* stop_condition, const MWString* const network_file_name_prefix, const size_t write_interval) const = 0;
};

#endif
