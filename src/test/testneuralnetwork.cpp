#include <iostream>
#include <float.h>

#include <mwglobal.h>
#include <mwlogger.h>

#include <mwstring.h>
#include <binfile.h>
#include <mwtimer.h>

#include <mwmathglobal.h>
#include <mwmatrix.h>
#include <vectormatrixmultithreadcalculator.h>

#include "neuralnetworkglobal.h"
#include "neuralnetwork.h"
#include "traininginformation.h"
#include "backpropgation.h"
#include "weightedsquarederrorfunction.h"

using namespace std;

#define PROGRAMM_NAME "testneuralnetwork"

int main(int argc, const char* argv[])
{
	if (argc < 7)
	{
		cout << "usage: " << argv[0] << " thread_num data_num hidden_neuron_num min_error network_file_name time_limit" << endl;
		return EXIT_FAILURE;
	}

	init_logger(PROGRAMM_NAME);

	MWString network_file_name(argv[5]);
	BinFile network_file(network_file_name);
	if (!network_file.open_for_write())
	{
		LOG_ERROR(_logger, "open file " << network_file_name << " for write error");
		return EXIT_FAILURE;
	}
	
	const size_t data_num = MWString(argv[2]).to_size_t(), hidden_neuron_num = MWString(argv[3]).to_size_t();
	LOG_INFO(_logger, "start, data_num = " << data_num);

	Vector<Matrix<Data> > p_vector_matrix(data_num * data_num, Matrix<Data>(2, 1)), t_vector_matrix(data_num * data_num, Matrix<Data>(1, 1)), test_p_vector_matrix(data_num * data_num, Matrix<Data>(2, 1)), test_t_vector_matrix(data_num * data_num, Matrix<Data>(1, 1)), a_minus_t_vector_matrix = test_t_vector_matrix;
	size_t i, j;
	for (i = 0; i < data_num; ++ i)
	{
		for (j = 0; j < data_num; ++ j)
		{
			t_vector_matrix[i * data_num + j]._data[0][0] = (p_vector_matrix[i * data_num + j]._data[0][0] = ((Data) (i + 1) / (data_num + 1) + 1)) * (p_vector_matrix[i * data_num + j]._data[1][0] = ((Data) (j + 1) / (data_num + 1) + 1));
			test_t_vector_matrix[i * data_num + j]._data[0][0] = (test_p_vector_matrix[i * data_num + j]._data[0][0] = ((Data) i / data_num + 1)) * (test_p_vector_matrix[i * data_num + j]._data[1][0] = ((Data) j / data_num + 1));
		}
	}

	LOG_TRACE(_logger, "t_vector_matrix = " << t_vector_matrix);
	LOG_TRACE(_logger, "p_vector_matrix = " << p_vector_matrix);
	LOG_TRACE(_logger, "test_t_vector_matrix = " << test_t_vector_matrix << ", test_p_vector_matrix = " << test_p_vector_matrix);

	const size_t thread_num = MWString(argv[1]).to_size_t();
	VectorMatrixMultiThreadCalculator::initialize(thread_num);
	MWMathFunction::initialize(thread_num);

	NeuralNetwork network(2, hidden_neuron_num, 1);
	double min_error = MWString(argv[4]).to_double(), time_limit = MWString(argv[6]).to_double(), test_error;
	TrainingInformation stop_condition(min_error, 1000, time_limit, MWTimer::get_time()), training_information;
	LOG_INFO(_logger, "training starts, network.get_neuron_num_vector() = " << network.get_neuron_num_vector() << ", using min_error = " << min_error << ", train_info dispalys every 1000 epochs");
	size_t epoch = 0;
	Vector<Vector<Matrix<Data> > > n_vector_vector_matrix, a_vector_vector_matrix;
	Vector<Matrix<Data> > a_end_vector_matrix;
	network.assign_output_structure(p_vector_matrix.size(), n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix);
	a_vector_vector_matrix[0] = test_p_vector_matrix;
	while (true)
	{
		training_information = BackPropgation::get_instance()->train(& network, p_vector_matrix, t_vector_matrix, WeightedSquaredErrorFunction::get_instance(), & stop_condition, 0.5);
		test_error = WeightedSquaredErrorFunction::get_instance()->get_error(network.assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix), test_t_vector_matrix, a_minus_t_vector_matrix);
		cout << "error = " << training_information.get_error() << ", test_error = " << test_error << ", epoch = " << (epoch += training_information.get_epoch()) << endl;
		LOG_INFO(_logger, "error = " << training_information.get_error() << ", test_error = " << test_error << ", epoch = " << epoch);
		if (training_information.is_satisfied(& stop_condition))
		{
			break;
		}
	}
	LOG_INFO(_logger, "training finished");

	MWMathFunction::finalize();
	VectorMatrixMultiThreadCalculator::finalize();

	LOG_INFO(_logger, "network.get_w_vector_matrix() = " << network.get_w_vector_matrix() << ", network.get_b_vector_matrix() = " << network.get_b_vector_matrix() << ", network.get_neuron_num_vector() = " << network.get_neuron_num_vector());

	vector<char> char_vector = network.get_serialized();
	if (!network_file.write(char_vector))
	{
		LOG_ERROR(_logger, "write char_vector to " << network_file_name << " error");
		return EXIT_FAILURE;
	}

	LOG_INFO(_logger, "writing finished");
	
	return EXIT_SUCCESS;
}
