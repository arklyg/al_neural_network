#include <iostream>
#include <float.h>

#include <mwglobal.h>
#include <mwlogger.h>

#include <mwstring.h>
#include <binfile.h>

#include <mwmathglobal.h>
#include <mwmatrix.h>
#include <vectormatrixmultithreadcalculator.h>

#include "neuralnetworkglobal.h"
#include "neuralnetwork.h"
#include "traininginformation.h"
#include "backpropgation.h"
#include "weightedsquarederrorfunction.h"

using namespace std;

#define PROGRAMM_NAME "testreloadneuralnetwork"

int main(int argv, const char* args[])
{
	if (argv < 2)
	{
		cout << "usage: " << args[0] << " network_file_name" << endl;
		return EXIT_FAILURE;
	}

	init_logger(PROGRAMM_NAME);

	MWString network_file_name(args[1]);
	BinFile network_file(network_file_name);
	if (!network_file.open_for_read())
	{
		LOG_ERROR(_logger, "open " << network_file_name << " for read error");
		return EXIT_FAILURE;
	}

	vector<char> char_vector = network_file.read();
	if (char_vector.size() == 0)
	{
		LOG_ERROR(_logger, "read 0");
		return EXIT_FAILURE;
	}
	network_file.close();

	NeuralNetwork network;
	network.get_instantiated(char_vector, 0);
	network.initialize_active_function();

	LOG_INFO(_logger, "network.get_w_vector_matrix() = " << network.get_w_vector_matrix() << ", network.get_b_vector_matrix() = " << network.get_b_vector_matrix() << ", network.get_neuron_num_vector() = " << network.get_neuron_num_vector());

	Vector<Vector<Matrix<Data> > > n_vector_vector_matrix, a_vector_vector_matrix;
	Vector<Matrix<Data> > a_end_vector_matrix;
	network.assign_output_structure(1, n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix);
	a_vector_vector_matrix[0] = Vector<Matrix<Data> >(1, Matrix<Data>(2, 1));
	while (true)
	{
		cout << "input first double x, (1 < x < 2): ";
		cin >> a_vector_vector_matrix[0][0]._data[0][0];
		cout << "input second double y, (1 < y < 2): ";
		cin >> a_vector_vector_matrix[0][0]._data[1][0];
		network.assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix);
		cout << "x * y, network result = " << a_end_vector_matrix[0]._data[0][0] << ", result should be " << (a_vector_vector_matrix[0][0]._data[0][0] * a_vector_vector_matrix[0][0]._data[1][0]) << ", error = " << (pow(a_end_vector_matrix[0]._data[0][0] - a_vector_vector_matrix[0][0]._data[0][0] * a_vector_vector_matrix[0][0]._data[1][0], 2)) << endl;
	}

	return EXIT_SUCCESS;
}
