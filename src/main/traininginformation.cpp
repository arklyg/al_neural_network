#include <mwtimer.h>

#include "traininginformation.h"

TrainingInformation::TrainingInformation(Data error, size_t epoch, double time)
	: _error(error)
	, _epoch(epoch)
	, _time(time)
{
}

void TrainingInformation::set_error(Data error)
{
	_error = error;
}

void TrainingInformation::set_epoch(size_t epoch)
{
	_epoch = epoch;
}

void TrainingInformation::set_time_limit(double time_limit)
{
	_time = time_limit;
}

void TrainingInformation::set_start_time(double start_time)
{
	_time = start_time;
}

void TrainingInformation::set_start_time()
{
	return set_start_time(MWTimer::get_time());
}

Data TrainingInformation::get_error() const
{
	return _error;
}

size_t TrainingInformation::get_epoch() const
{
	return _epoch;
}

double TrainingInformation::get_time_limit() const
{
	return _time;
}

double TrainingInformation::get_start_time() const
{
	return _time;
}

size_t TrainingInformation::age()
{
	return ++ _epoch;
}

bool TrainingInformation::is_satisfied(const TrainingInformation* information) const
{
	if (_error < information->get_error())
	{
		LOG_INFO(_logger, "(_error = " << _error << ") < (target_error = " << information->get_error() << "), satisfied");
		return true;
	}
	// epoch 为零则不作为满足条件
	if (information->get_epoch() > 0 && _epoch >= information->get_epoch())
	{
		LOG_INFO(_logger, "(_epoch = " << _epoch << ") >= (target_epoch = " << information->get_epoch() << "), satisfied");
		return true;
	}
	// time_limit 为零则不作为满足条件
	if (information->get_time_limit() > 0 && this->get_cost_time() > information->get_time_limit())
	{
		LOG_INFO(_logger, "(cost_time = " << this->get_cost_time() << ") > (_time_limit = " << information->get_time_limit() << "), satisfied");
		return true;
	}

	return false;
}

double TrainingInformation::get_cost_time() const
{
	return MWTimer::endure(_time);
}
