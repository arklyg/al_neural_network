#include <logsigfunction.h>
#include <linearfunction.h>
#include <randomfunction.h>
#include <mwmatrixhelper.h>
#include <vectormatrixmultithreadcalculator.h>

#include "neuralnetwork.h"

void NeuralNetwork::add_serializing_sequence()
{
	add_serializing_sequence_data(& _neuron_num_vector);
	add_serializing_sequence_data(& _w_vector_matrix);
	add_serializing_sequence_data(& _b_vector_matrix);
}

NeuralNetwork::NeuralNetwork(size_t input_num, size_t hidden_num, size_t output_num)
	: _neuron_num_vector(Vector<size_t>(NETWORK_LAYER_NUM + 1))
	, _active_function_vector(vector<MWDerivableMathFunction*>(NETWORK_LAYER_NUM))
	, _w_vector_matrix(Vector<Matrix<Data> >(NETWORK_LAYER_NUM))
	, _b_vector_matrix(Vector<Matrix<Data> >(NETWORK_LAYER_NUM))
{
	init(input_num, hidden_num, output_num);
}

void NeuralNetwork::init(size_t input_num, size_t hidden_num, size_t output_num)
{
	_neuron_num_vector[0] = input_num;
	_neuron_num_vector[1] = hidden_num;
	_neuron_num_vector[2] = output_num;

	Vector<Data> w_b_random_min_max(2);
	w_b_random_min_max[0] = W_B_RANDOM_MIN;
	w_b_random_min_max[1] = W_B_RANDOM_MAX;

	for (size_t i_layer = 0, i, j; i_layer < NETWORK_LAYER_NUM; ++ i_layer)
	{
		_w_vector_matrix[i_layer] = Matrix<Data>(_neuron_num_vector[i_layer + 1], _neuron_num_vector[i_layer]);
		_b_vector_matrix[i_layer] = Matrix<Data>(_neuron_num_vector[i_layer + 1], 1);

		// 赋随机值
		for (i = 0; i < _neuron_num_vector[i_layer + 1]; ++ i)
		{
#ifdef _COMPILE_MODE_TEST_
			_b_vector_matrix[i_layer]._data[i][0] = 0;
#else
			_b_vector_matrix[i_layer]._data[i][0] = RandomFunction::get_instance()->get_value(w_b_random_min_max)[0];
#endif
			for (j = 0; j < _neuron_num_vector[i_layer]; ++ j)
			{
#ifdef _COMPILE_MODE_TEST_
				_w_vector_matrix[i_layer]._data[i][j] = 0;
#else
				_w_vector_matrix[i_layer]._data[i][j] = RandomFunction::get_instance()->get_value(w_b_random_min_max)[0];
#endif
			}
		}
	}

	_active_function_vector[0] = LogsigFunction::get_instance();
	_active_function_vector[1] = LinearFunction::get_instance();
}

void NeuralNetwork::set_active_function(MWDerivableMathFunction* active_function_arr[])
{
	for (size_t i = 0; i < NETWORK_LAYER_NUM; ++ i)
	{
		_active_function_vector[i] = active_function_arr[i];
	}
}

void NeuralNetwork::initialize_active_function()
{
	MWDerivableMathFunction* functions[2];
	functions[0] = LogsigFunction::get_instance();
	functions[1] = LinearFunction::get_instance();
	set_active_function(functions);
}

const MWDerivableMathFunction* NeuralNetwork::get_active_function(size_t i_layer)
{
	if (i_layer >= NETWORK_LAYER_NUM)
	{
		LOG_ERROR(_logger, "(i_layer = " << i_layer << ") >= (NETWORK_LAYER_NUM = " << NETWORK_LAYER_NUM << "), can not get_active_function, return NULL");
		return NULL;
	}

	return _active_function_vector[i_layer];
}

void NeuralNetwork::assign_output_structure(size_t vector_size, Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix, Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix, Vector<Matrix<Data> > &a_end_vector_matrix) const
{
	n_vector_vector_matrix = Vector<Vector<Matrix<Data> > >(NETWORK_LAYER_NUM);
	a_vector_vector_matrix = Vector<Vector<Matrix<Data> > >(NETWORK_LAYER_NUM);
	for (size_t i_layer = 0; i_layer < NETWORK_LAYER_NUM; ++ i_layer)
	{
		n_vector_vector_matrix[i_layer] = Vector<Matrix<Data> >(vector_size, Matrix<Data>(_neuron_num_vector[i_layer + 1], 1));
		(i_layer + 1 < NETWORK_LAYER_NUM ? a_vector_vector_matrix[i_layer + 1] : a_end_vector_matrix) = n_vector_vector_matrix[i_layer];
	}

	return;
}

Vector<Matrix<Data> > &NeuralNetwork::assign_output(Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix, Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix, Vector<Matrix<Data> > &a_end_vector_matrix, const Vector<Matrix<Data> > &w_vector_matrix, const Vector<Matrix<Data> > &b_vector_matrix) const
{
	for (size_t i_layer = 0; i_layer < NETWORK_LAYER_NUM; ++ i_layer)
	{
		//a_vector_vector_matrix[i_layer + 1] = _active_function_vector[i_layer]->_value_vector_matrix(n_vector_vector_matrix[i_layer] = w_vector_matrix[i_layer] * a_vector_vector_matrix[i_layer] + b_vector_matrix[i_layer]);
		_active_function_vector[i_layer]->assign_value_vector_matrix_multi_thread(VectorMatrixMultiThreadCalculator::assign_plus(VectorMatrixMultiThreadCalculator::assign_times(w_vector_matrix[i_layer], a_vector_vector_matrix[i_layer], n_vector_vector_matrix[i_layer]), b_vector_matrix[i_layer], n_vector_vector_matrix[i_layer]), (i_layer + 1 == NETWORK_LAYER_NUM) ? a_end_vector_matrix : a_vector_vector_matrix[i_layer + 1]);
	}

	return a_end_vector_matrix;
}

Vector<Matrix<Data> > &NeuralNetwork::assign_output(Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix, Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix, Vector<Matrix<Data> > &a_end_vector_matrix) const
{
	return assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix, _w_vector_matrix, _b_vector_matrix);
}

Vector<Matrix<Data> > NeuralNetwork::get_output(const Vector<Matrix<Data> > &p) const
{
	Vector<Matrix<Data> > a_end_vector_matrix;
	Vector<Vector<Matrix<Data> > > n_vector_vector_matrix, a_vector_vector_matrix;
	assign_output_structure(p.size(), n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix);

	a_vector_vector_matrix[0] = p;
	assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix);
	return a_end_vector_matrix;
}

const Vector<size_t> &NeuralNetwork::get_neuron_num_vector() const
{
	return _neuron_num_vector;
}

const Vector<Matrix<Data> > &NeuralNetwork::get_w_vector_matrix() const
{
	return _w_vector_matrix;
}

const Vector<Matrix<Data> > &NeuralNetwork::get_b_vector_matrix() const
{
	return _b_vector_matrix;
}

Vector<Matrix<Data> > &NeuralNetwork::assign_w_vector_matrix(Vector<Matrix<Data> > &w_vector_matrix) const
{
	return assign(_w_vector_matrix, w_vector_matrix);
}

Vector<Matrix<Data> > &NeuralNetwork::assign_b_vector_matrix(Vector<Matrix<Data> > &b_vector_matrix) const
{
	return assign(_b_vector_matrix, b_vector_matrix);
}

void NeuralNetwork::set_w_vector_matrix(const Vector<Matrix<Data> > &w_vector_matrix)
{
	_w_vector_matrix = w_vector_matrix;
}
void NeuralNetwork::set_b_vector_matrix(const Vector<Matrix<Data> > &b_vector_matrix)
{
	_b_vector_matrix = b_vector_matrix;
}

size_t NeuralNetwork::get_layer_num() const
{
	return NETWORK_LAYER_NUM;
}
