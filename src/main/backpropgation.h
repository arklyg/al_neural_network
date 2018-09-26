#ifndef _BACK_PROPGATION_H_
#define _BACK_PROPGATION_H_

#include <mwsingleton.h>

#include "trainingstrategy.h"

#define BACK_PROPGATION_DELTA_ERROR_MIN 0.00000001
#define BACK_PROPGATION_DELTA_ERROR_ITERATION_MIN 3

class Backpropgation;

class BackPropgation : public TrainingStrategy, public MWSingleton<BackPropgation>
{
private:
	int save_network(const MWString & network_file_name, NeuralNetwork* network) const;

public:
	virtual TrainingInformation train(NeuralNetwork* network, const MWVector<Matrix<Data> > &p_vector_matrix, const MWVector<Matrix<Data> > &t_vector_matrix, const CanOperateWeightsErrorFunction* error_function, const TrainingInformation* stop_condition, const MWString* const network_file_name_prefix = NULL, const size_t write_interval = 0) const;
};

#endif
