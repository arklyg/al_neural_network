#include <mwglobal.h>

#include <binfile.h>

#include <math.h>
#include <mwmatrixhelper.h>

#include <mwderivablemathfunction.h>
#include <vectormatrixmultithreadcalculator.h>

#include "neuralnetwork.h"
#include "canoperateweightserrorfunction.h"

#include "backpropgation.h"

TrainingInformation BackPropgation::train(NeuralNetwork* network, const MWVector<Matrix<Data> > &p_vector_matrix, const MWVector<Matrix<Data> > &t_vector_matrix, const CanOperateWeightsErrorFunction* error_function, const TrainingInformation* stop_condition, const MWString* const network_file_name_prefix, const size_t write_interval) const
{
	TrainingInformation training_information;
	const size_t vector_size(p_vector_matrix.size()), layer_num(network->get_layer_num());

	// 先行判断
	if (vector_size == 0)
	{
		LOG_WARN(_logger, "vector_size == 0, return");
		return training_information;
	}

	// 初始化各中间数组，结果数组
	Vector<Matrix<Data> > w_vector_matrix = network->get_w_vector_matrix(), delta_w_vector_matrix = w_vector_matrix, final_w_vector_matrix = w_vector_matrix, b_vector_matrix = network->get_b_vector_matrix(), delta_b_vector_matrix = b_vector_matrix, final_b_vector_matrix = b_vector_matrix, a_end_vector_matrix;

	Vector<Vector<Matrix<Data> > > n_vector_vector_matrix, a_vector_vector_matrix;
	network->assign_output_structure(vector_size, n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix);
	Vector<Matrix<Data> > a_minus_t_vector_matrix = a_end_vector_matrix, a_minus_t_vector_matrix_mid = a_end_vector_matrix;

	Vector<Vector<Matrix<Data> > > f_vector_vector_matrix = n_vector_vector_matrix, f_times_w_vector_vector_matrix(layer_num - 1), s_vector_vector_matrix(layer_num), delta_w_vector_vector_matrix(layer_num);

	size_t i_layer;
	for (i_layer = 0; i_layer < layer_num; ++ i_layer)
	{
		if (i_layer + 1 < layer_num)
		{
			f_times_w_vector_vector_matrix[i_layer] = Vector<Matrix<Data> >(vector_size, Matrix<Data>(w_vector_matrix[i_layer + 1].get_column_num(), w_vector_matrix[i_layer + 1].get_row_num()));
		}
		s_vector_vector_matrix[i_layer] = Vector<Matrix<Data> >(vector_size, b_vector_matrix[i_layer]);
		delta_w_vector_vector_matrix[i_layer] = Vector<Matrix<Data> >(vector_size, w_vector_matrix[i_layer]);
	}

	size_t iteration_count;

	double learning_rate_a, learning_rate_c = 1, learning_rate_d, learning_rate_b, error_a, error_c, error_d, error_b, cost_time(0), last_write_time(0);

	a_vector_vector_matrix[0] = p_vector_matrix;
	training_information.set_start_time();
	while (true)
	{
		// 保存network
		if (network_file_name_prefix != NULL && cost_time - last_write_time > write_interval)
		{
			if (save_network(MWString((* network_file_name_prefix) + MWString::to_string(training_information.get_epoch()) + "-" + MWString::to_string(error_c)), network) < 0)
			{
				break;
			}
			/*
			const MWString network_file_name((* network_file_name_prefix) + MWString::to_string(training_information.get_epoch()) + "-" + MWString::to_string(error_c));
			BinFile network_file(network_file_name);
			if (!network_file.open_for_write())
			{
				LOG_ERROR(_logger, "open file " << network_file_name << " for write error");
				break;
			}
			if (!network_file.write(network->get_serialized()))
			{
				LOG_ERROR(_logger, "write char_vector to " << network_file_name << " error");
				break;
			}
			network_file.close();
			LOG_INFO(_logger, "network saved to " << network_file_name);
			*/

			last_write_time = cost_time;
		}

		// 正向传播，存储各层的神经元函数输入及神经元输出
		network->assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix);
		LOG_TRACE(_logger, "w_vector_matrix = " << network->get_w_vector_matrix() << ", b_vector_matrix = " << network->get_b_vector_matrix() << ", p_vector_matrix = " << p_vector_matrix << ", n_vector_vector_matrix = " << n_vector_vector_matrix << ", a_vector_vector_matrix = " << a_vector_vector_matrix);

		// 是否满足停止条件
		training_information.set_error(error_function->get_error(a_end_vector_matrix, t_vector_matrix, a_minus_t_vector_matrix, a_minus_t_vector_matrix_mid));
		LOG_TRACE(_logger, "training_information.get_error() = " << training_information.get_error());
		if (training_information.is_satisfied(stop_condition))
		{
			LOG_DEBUG(_logger, "conditions satisfied, break");
			break;
		}

		// s种子
		//s_vector_vector_matrix.push_back(a_vector_vector_matrix.back() - t_vector_matrix);
		
		// w
		//w_vector_matrix = network->get_w_vector_matrix();
		network->assign_w_vector_matrix(w_vector_matrix);
		// w种子: I矩阵
		//w_vector_matrix.push_back(Matrix<Data>(Vector<Data>(t_vector_matrix[0].get_row_num(), 1)));

		// 反向传播
		// 最后一层的雅克比矩阵
		network->get_active_function(layer_num - 1)->assign_derivative_vector_matrix_multi_thread(n_vector_vector_matrix[layer_num - 1], f_vector_vector_matrix[layer_num - 1]);
		// 最后一层的敏感性
		VectorMatrixMultiThreadCalculator::assign_diagonaled_times(f_vector_vector_matrix[layer_num - 1], VectorMatrixMultiThreadCalculator::assign_minus(a_end_vector_matrix, t_vector_matrix, a_minus_t_vector_matrix), s_vector_vector_matrix[layer_num - 1]);

		// 从倒数第二层开始
		for (i_layer = layer_num - 1; i_layer > 0; -- i_layer)
		{
			// 获取本层神经元函数的雅克比矩阵
			//f_vector_matrix = network->get_active_function(i_layer - 1)->get_derivative_vector_diagonal_matrix(n_vector_vector_matrix[i_layer - 1]);
			network->get_active_function(i_layer - 1)->assign_derivative_vector_matrix_multi_thread(n_vector_vector_matrix[i_layer - 1], f_vector_vector_matrix[i_layer - 1]);
			// 计算本层敏感性
			//s_vector_vector_matrix[i_layer - 1] = f_vector_matrix * w_vector_matrix[i_layer].t() * s_vector_vector_matrix[i_layer];
			VectorMatrixMultiThreadCalculator::assign_times(VectorMatrixMultiThreadCalculator::assign_diagonaled_times_transformed(f_vector_vector_matrix[i_layer - 1], w_vector_matrix[i_layer], f_times_w_vector_vector_matrix[i_layer - 1]), s_vector_vector_matrix[i_layer], s_vector_vector_matrix[i_layer - 1]);
		}

		// 获取w，b的改变量
		LOG_TRACE(_logger, "s_vector_vector_matrix = " << s_vector_vector_matrix << ", a_vector_vector_matrix.t() = " << a_vector_vector_matrix.t());
		//delta_w_vector_vector_matrix = - (s_vector_vector_matrix * a_vector_vector_matrix.t());
		VectorMatrixMultiThreadCalculator::assign_times_transformed(s_vector_vector_matrix, a_vector_vector_matrix, delta_w_vector_vector_matrix);
		//delta_b_vector_vector_matrix = - s_vector_vector_matrix;
		//s_vector_vector_matrix.assign(delta_b_vector_vector_matrix);
		LOG_TRACE(_logger, "delta_w_vector_vector_matrix(should be minus) = " << delta_w_vector_vector_matrix << ", delta_b_vector_vector_matrix(should be minus) = " << s_vector_vector_matrix);

		// 获取w, b改变量的整体值
		//delta_w_vector_matrix = error_function->assign_global_weight(delta_w_vector_vector_matrix);
		error_function->assign_global_weight_multi_thread(delta_w_vector_vector_matrix, delta_w_vector_matrix);
		//delta_b_vector_matrix = error_function->assign_global_weight(delta_b_vector_vector_matrix);
		error_function->assign_global_weight_multi_thread(s_vector_vector_matrix, delta_b_vector_matrix);

		// 获取当前值
		//w_vector_matrix.pop_back();
		//b_vector_matrix = network->get_b_vector_matrix();
		network->assign_b_vector_matrix(b_vector_matrix);

		// 寻找共轭点
		learning_rate_a = 0;
		error_a = error_function->get_error(network->assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix, assign_plus(assign_times(delta_w_vector_matrix, - learning_rate_a, final_w_vector_matrix), w_vector_matrix, final_w_vector_matrix), assign_plus(assign_times(delta_b_vector_matrix, - learning_rate_a, final_b_vector_matrix), b_vector_matrix, final_b_vector_matrix)), t_vector_matrix, a_minus_t_vector_matrix, a_minus_t_vector_matrix_mid);
		//error_c = error_function->get_error(network->get_output(p_vector_matrix, n_vector_vector_matrix, a_vector_vector_matrix, w_vector_matrix + delta_w_vector_matrix * learning_rate_c, b_vector_matrix + delta_b_vector_matrix * learning_rate_c), t_vector_matrix);
		error_c = error_function->get_error(network->assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix, assign_plus(assign_times(delta_w_vector_matrix, - learning_rate_c, final_w_vector_matrix), w_vector_matrix, final_w_vector_matrix), assign_plus(assign_times(delta_b_vector_matrix, - learning_rate_c, final_b_vector_matrix), b_vector_matrix, final_b_vector_matrix)), t_vector_matrix, a_minus_t_vector_matrix, a_minus_t_vector_matrix_mid);
		LOG_DEBUG(_logger, "error_a = " << error_a << ", error_c = " << error_c << ", learning_rate_a = " << learning_rate_a << ", learning_rate_c = " << learning_rate_c);

		if (error_a < error_c) // . '
		{
			do
			{
				learning_rate_b = learning_rate_c;
				error_b = error_c;
				learning_rate_c = learning_rate_a + (learning_rate_b - learning_rate_a) * (1 - MW_MATH_GOLDEN_RATIO);
				//error_c = error_function->get_error(network->get_output(p_vector_matrix, n_vector_vector_matrix, a_vector_vector_matrix, w_vector_matrix + delta_w_vector_matrix * learning_rate_c, b_vector_matrix + delta_b_vector_matrix * learning_rate_c), t_vector_matrix);
				error_c = error_function->get_error(network->assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix, assign_plus(assign_times(delta_w_vector_matrix, - learning_rate_c, final_w_vector_matrix), w_vector_matrix, final_w_vector_matrix), assign_plus(assign_times(delta_b_vector_matrix, - learning_rate_c, final_b_vector_matrix), b_vector_matrix, final_b_vector_matrix)), t_vector_matrix, a_minus_t_vector_matrix, a_minus_t_vector_matrix_mid);
			} while (error_a < error_c); // . '
			// ' . '
			LOG_DEBUG(_logger, "error_a = " << error_a << ", error_c = " << error_c << ", error_b = " << error_b << ", learning_rate_a = " << learning_rate_a << ", learning_rate_c = " << learning_rate_c << ", learning_rate_b = " << learning_rate_b);

			if (learning_rate_c == learning_rate_a)
			{
				LOG_DEBUG(_logger, "learning_rate_c == " << learning_rate_a << ", network reaches its limit, break.");
				break;
			}
		}
		else // ' .
		{
			learning_rate_b = learning_rate_a + (learning_rate_c - learning_rate_a) / (1 - MW_MATH_GOLDEN_RATIO);
			//error_b = error_function->get_error(network->get_output(p_vector_matrix, n_vector_vector_matrix, a_vector_vector_matrix, w_vector_matrix + delta_w_vector_matrix * learning_rate_b, b_vector_matrix + delta_b_vector_matrix * learning_rate_b), t_vector_matrix);
			error_b = error_function->get_error(network->assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix, assign_plus(assign_times(delta_w_vector_matrix, - learning_rate_b, final_w_vector_matrix), w_vector_matrix, final_w_vector_matrix), assign_plus(assign_times(delta_b_vector_matrix, - learning_rate_b, final_b_vector_matrix), b_vector_matrix, final_b_vector_matrix)), t_vector_matrix, a_minus_t_vector_matrix, a_minus_t_vector_matrix_mid);

			while (error_c > error_b) // ' .
			{
				learning_rate_a = learning_rate_c;
				error_a = error_c;
				learning_rate_c = learning_rate_b;
				error_c = error_b;

				learning_rate_b = learning_rate_a + (learning_rate_c - learning_rate_a) / (1 - MW_MATH_GOLDEN_RATIO);
				//error_b = error_function->get_error(network->get_output(p_vector_matrix, n_vector_vector_matrix, a_vector_vector_matrix, w_vector_matrix + delta_w_vector_matrix * learning_rate_b, b_vector_matrix + delta_b_vector_matrix * learning_rate_b), t_vector_matrix);
				error_b = error_function->get_error(network->assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix, assign_plus(assign_times(delta_w_vector_matrix, - learning_rate_b, final_w_vector_matrix), w_vector_matrix, final_w_vector_matrix), assign_plus(assign_times(delta_b_vector_matrix, - learning_rate_b, final_b_vector_matrix), b_vector_matrix, final_b_vector_matrix)), t_vector_matrix, a_minus_t_vector_matrix, a_minus_t_vector_matrix_mid);
			}
			// ' . '
			LOG_DEBUG(_logger, "error_a = " << error_a << ", error_c = " << error_c << ", error_b = " << error_b << ", learning_rate_a = " << learning_rate_a << ", learning_rate_c = " << learning_rate_c << ", learning_rate_b = " << learning_rate_b);
		}

		learning_rate_d = learning_rate_a + (learning_rate_b - learning_rate_a) * MW_MATH_GOLDEN_RATIO;
		//error_d = error_function->get_error(network->get_output(p_vector_matrix, n_vector_vector_matrix, a_vector_vector_matrix, w_vector_matrix + delta_w_vector_matrix * learning_rate_d, b_vector_matrix + delta_b_vector_matrix * learning_rate_d), t_vector_matrix);
		error_d = error_function->get_error(network->assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix, assign_plus(assign_times(delta_w_vector_matrix, - learning_rate_d, final_w_vector_matrix), w_vector_matrix, final_w_vector_matrix), assign_plus(assign_times(delta_b_vector_matrix, - learning_rate_d, final_b_vector_matrix), b_vector_matrix, final_b_vector_matrix)), t_vector_matrix, a_minus_t_vector_matrix, a_minus_t_vector_matrix_mid);
		LOG_DEBUG(_logger, "error_a = " << error_a << ", error_c = " << error_c << ", error_d = " << error_d << ", error_b = " << error_b << ", learning_rate_a = " << learning_rate_a << ", learning_rate_c = " << learning_rate_c << ", learning_rate_d = " << learning_rate_d << ", learning_rate_b = " << learning_rate_b);
		iteration_count = 0;

		// a 39 c 22 d 39 b
		//while (error_c != error_d && (iteration_count ++ < BACK_PROPGATION_DELTA_ERROR_ITERATION_MIN || fabs(error_c - error_d) / error_c > BACK_PROPGATION_DELTA_ERROR_MIN) && learning_rate_c < learning_rate_d)
		while (error_c != error_d && learning_rate_c < learning_rate_d)
		{
			++ iteration_count;

			if (error_c < error_d)
			{
				learning_rate_b = learning_rate_d;
				error_b = error_d;
				learning_rate_d = learning_rate_c;
				error_d = error_c;
				learning_rate_c = learning_rate_a + (learning_rate_b - learning_rate_a) * (1 - MW_MATH_GOLDEN_RATIO);
				//error_c = error_function->get_error(network->get_output(p_vector_matrix, n_vector_vector_matrix, a_vector_vector_matrix, w_vector_matrix + delta_w_vector_matrix * learning_rate_c, b_vector_matrix + delta_b_vector_matrix * learning_rate_c), t_vector_matrix);
				error_c = error_function->get_error(network->assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix, assign_plus(assign_times(delta_w_vector_matrix, - learning_rate_c, final_w_vector_matrix), w_vector_matrix, final_w_vector_matrix), assign_plus(assign_times(delta_b_vector_matrix, - learning_rate_c, final_b_vector_matrix), b_vector_matrix, final_b_vector_matrix)), t_vector_matrix, a_minus_t_vector_matrix, a_minus_t_vector_matrix_mid);
			}
			else
			{
				learning_rate_a = learning_rate_c;
				error_a = error_c;
				learning_rate_c = learning_rate_d;
				error_c = error_d;
				learning_rate_d = learning_rate_a + (learning_rate_b - learning_rate_a) * MW_MATH_GOLDEN_RATIO;
				//error_d = error_function->get_error(network->get_output(p_vector_matrix, n_vector_vector_matrix, a_vector_vector_matrix, w_vector_matrix + delta_w_vector_matrix * learning_rate_d, b_vector_matrix + delta_b_vector_matrix * learning_rate_d), t_vector_matrix);
				error_d = error_function->get_error(network->assign_output(n_vector_vector_matrix, a_vector_vector_matrix, a_end_vector_matrix, assign_plus(assign_times(delta_w_vector_matrix, - learning_rate_d, final_w_vector_matrix), w_vector_matrix, final_w_vector_matrix), assign_plus(assign_times(delta_b_vector_matrix, - learning_rate_d, final_b_vector_matrix), b_vector_matrix, final_b_vector_matrix)), t_vector_matrix, a_minus_t_vector_matrix, a_minus_t_vector_matrix_mid);
			}
			LOG_DEBUG(_logger, "error_a = " << error_a << ", error_c = " << error_c << ", error_d = " << error_d << ", error_b = " << error_b << ", learning_rate_a = " << learning_rate_a << ", learning_rate_c = " << learning_rate_c << ", learning_rate_d = " << learning_rate_d << ", learning_rate_b = " << learning_rate_b);
		}

		LOG_TRACE(_logger, "delta_w_vector_matrix = " << delta_w_vector_matrix << ", delta_b_vector_matrix = " << delta_b_vector_matrix << ", learning_rate = " << learning_rate_c);
		// 更新W, b
		network->set_w_vector_matrix(assign_plus(w_vector_matrix, assign_times(delta_w_vector_matrix, - learning_rate_c, delta_w_vector_matrix), w_vector_matrix));
		//network->set_b_vector_matrix(b_vector_matrix + delta_b_vector_matrix * learning_rate_c);
		network->set_b_vector_matrix(assign_plus(b_vector_matrix, assign_times(delta_b_vector_matrix, - learning_rate_c, delta_b_vector_matrix), b_vector_matrix));

		// ++ epoch
		training_information.age();

		cost_time = training_information.get_cost_time();
		LOG_INFO(_logger, "epoch=" << training_information.get_epoch() << ",mse=" << error_c << ",lr=" << learning_rate_c << ", ic=" << iteration_count << ",cost_time=" << cost_time << ",left_time=" << (stop_condition->get_time_limit() - cost_time));

		if (iteration_count == 0)
		{
			LOG_DEBUG(_logger, "iteration_count == 0, break");
			break;
		}

	}

	if (network_file_name_prefix != NULL)
	{
		save_network(MWString((* network_file_name_prefix) + MWString::to_string(training_information.get_epoch()) + "-" + MWString::to_string(error_c)), network);
	}

	return training_information;
}

int BackPropgation::save_network(const MWString & network_file_name, NeuralNetwork* network) const
{
	BinFile network_file(network_file_name);
	if (!network_file.open_for_write())
	{
		LOG_ERROR(_logger, "open file " << network_file_name << " for write error");
		return -1;
	}
	if (!network_file.write(network->get_serialized()))
	{
		LOG_ERROR(_logger, "write char_vector to " << network_file_name << " error");
		network_file.close();
		return -1;
	}
	network_file.close();
	LOG_INFO(_logger, "network saved to " << network_file_name);
	return 0;
}
