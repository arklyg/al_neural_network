#ifndef _TRAINING_INFORMATION_
#define _TRAINING_INFORMATION_

#include <float.h>

#include <mwmatrix.h>

#include "neuralnetworkglobal.h"

#define TRAINING_INFORMATION_DEFAULT_ERROR 0.01

class TrainingInformation
{
private:
	Data _error;
	size_t _epoch;
	double _time; // 做info时是start_time 做condition时是time_limit

public:
	TrainingInformation(Data error = TRAINING_INFORMATION_DEFAULT_ERROR, size_t epoch = 0, double time = 0);

	void set_error(Data error);
	void set_epoch(size_t epoch);
	void set_time_limit(double time_limit);
	void set_start_time(double start_time);
	void set_start_time();

	Data get_error() const;
	size_t get_epoch() const;
	double get_time_limit() const;
	double get_start_time() const;
	double get_cost_time() const;

	size_t age();

	bool is_satisfied(const TrainingInformation* information) const;
};

#endif
