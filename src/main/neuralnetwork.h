#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#define W_B_RANDOM_MIN -1
#define W_B_RANDOM_MAX 1

#define NETWORK_LAYER_NUM 2

#define DEFAULT_INPUT_NUM 1
#define DEFAULT_HIDDEN_NUM 2
#define DEFAULT_OUTPUT_NUM 1

#include <mwmatrix.h>

#include <mwserializable.h>

#include "neuralnetworkglobal.h"

class MWDerivableMathFunction;

class NeuralNetwork : public MWSerializable
{
private:
	Vector<size_t> _neuron_num_vector;
	vector<MWDerivableMathFunction*> _active_function_vector;
	Vector<Matrix<Data> > _w_vector_matrix;
	Vector<Matrix<Data> > _b_vector_matrix;

protected:
	virtual void add_serializing_sequence();

public:

	NeuralNetwork(size_t input_num = DEFAULT_INPUT_NUM, size_t hidden_num = DEFAULT_HIDDEN_NUM, size_t output_num = DEFAULT_OUTPUT_NUM);
	void init(size_t input_num = DEFAULT_INPUT_NUM, size_t hidden_num = DEFAULT_HIDDEN_NUM, size_t output_num = DEFAULT_OUTPUT_NUM);

	void set_active_function(MWDerivableMathFunction* active_function_arr[]);
	void initialize_active_function();
	const MWDerivableMathFunction* get_active_function(size_t layer_num);

	const Vector<size_t> &get_neuron_num_vector() const;
	const Vector<Matrix<Data> > &get_w_vector_matrix() const;
	const Vector<Matrix<Data> > &get_b_vector_matrix() const;

	Vector<Matrix<Data> > &assign_w_vector_matrix(Vector<Matrix<Data> > &w_vector_matrix) const;
	Vector<Matrix<Data> > &assign_b_vector_matrix(Vector<Matrix<Data> > &b_vector_matrix) const;

	void assign_output_structure(size_t vector_size, Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix, Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix, Vector<Matrix<Data> > &a_end_vector_matrix) const;
	Vector<Matrix<Data> > &assign_output(Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix, Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix, Vector<Matrix<Data> > &a_end_vector_matrix, const Vector<Matrix<Data> > &w_vector_matrix, const Vector<Matrix<Data> > &b_vector_matrix) const;
	Vector<Matrix<Data> > &assign_output(Vector<Vector<Matrix<Data> > > &n_vector_vector_matrix, Vector<Vector<Matrix<Data> > > &a_vector_vector_matrix, Vector<Matrix<Data> > &a_end_vector_matrix) const;
	Vector<Matrix<Data> > get_output(const Vector<Matrix<Data> > &p) const;

	void set_w_vector_matrix(const Vector<Matrix<Data> > &w);
	void set_b_vector_matrix(const Vector<Matrix<Data> > &b);

	size_t get_layer_num() const;
};

#endif
