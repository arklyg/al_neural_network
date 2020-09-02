// Micro Wave Neural Network
// Copyright (c) 2015-2020, Ark Lee
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// You must obey the GNU General Public License in all respects for
// all of the code used.  If you modify file(s) with this exception, 
// you may extend this exception to your version of the file(s), but 
// you are not obligated to do so.  If you do not wish to do so, delete 
// this exception statement from your version. If you delete this exception 
// statement from all source files in the program, then also delete it here.
//
// Contact:  Ark Lee <arklee@houduan.online>
//           Beijing, China


#include <mwglobal.h>

#include <binfile.h>

#include <math.h>
#include <mwmatrixhelper.h>

#include <mwderivablemathfunction.h>
#include <vectormatrixmultithreadcalculator.h>

#include "neuralnetwork.h"
#include "canoperateweightserrorfunction.h"

#include "backpropgation.h"

TrainingInformation BackPropgation::Train(NeuralNetwork *network,
                                          const MWVector<Matrix<Data> > &p_vector_matrix,
                                          const MWVector<Matrix<Data> > &t_vector_matrix,
                                          const CanOperateWeightsErrorFunction *error_function,
                                          const TrainingInformation *stop_condition,
                                          const MWString *const network_file_name_prefix,
                                          const size_t write_interval) const {
    TrainingInformation training_information;
    const size_t vector_size(p_vector_matrix.size()),
                 layer_num(network->GetLayerNum());

    // 先行判断
    if (vector_size == 0) {
        LOGWARN("vector_size == 0, return");
        return training_information;
    }

    // 初始化各中间数组，结果数组
    Vector<Matrix<Data> > w_vector_matrix = network->GetWVectorMatrix(),
                          delta_w_vector_matrix = w_vector_matrix,
                          final_w_vector_matrix = w_vector_matrix,
                          b_vector_matrix = network->GetBVectorMatrix(),
                          delta_b_vector_matrix = b_vector_matrix,
                          final_b_vector_matrix = b_vector_matrix, 
                          a_end_vector_matrix;

    Vector<Vector<Matrix<Data> > > n_vector_vector_matrix,
                                   a_vector_vector_matrix;
    network->AssignOutputStructure(vector_size,
                                   n_vector_vector_matrix,
                                   a_vector_vector_matrix,
                                   a_end_vector_matrix);
    Vector<Matrix<Data> > a_minus_t_vector_matrix = a_end_vector_matrix,
                          a_minus_t_vector_matrix_mid = a_end_vector_matrix;

    Vector<Vector<Matrix<Data> > > f_vector_vector_matrix = n_vector_vector_matrix,
                                   f_times_w_vector_vector_matrix(layer_num - 1),
                                   s_vector_vector_matrix(layer_num),
                                   delta_w_vector_vector_matrix(layer_num);

    size_t i_layer;
    for (i_layer = 0; i_layer < layer_num; ++ i_layer) {
        if (i_layer + 1 < layer_num) {
            f_times_w_vector_vector_matrix[i_layer] = 
                Vector<Matrix<Data> >(vector_size,
                                      Matrix<Data>(
                                          w_vector_matrix[i_layer + 1].GetColumnNum(),
                                          w_vector_matrix[i_layer + 1].GetRowNum()));
        }
        s_vector_vector_matrix[i_layer] = Vector<Matrix<Data> >(vector_size, b_vector_matrix[i_layer]);
        delta_w_vector_vector_matrix[i_layer] = Vector<Matrix<Data> >(vector_size, w_vector_matrix[i_layer]);
    }

    size_t iteration_count;
    double learning_rate_a,
           learning_rate_c = 1,
           learning_rate_d,
           learning_rate_b,
           error_a,
           error_c,
           error_d,
           error_b,
           cost_time = 0,
           last_write_time = 0;

    a_vector_vector_matrix[0] = p_vector_matrix;
    training_information.SetStartTime();
    while (true) {
        // 保存network
        if (network_file_name_prefix != NULL && cost_time - last_write_time > write_interval) {
            if (SaveNetwork(MWString((*network_file_name_prefix) + 
                                      MWString::ToString(training_information.GetEpoch()) + "-" + 
                                      MWString::ToString(error_c)),
                            network) < 0) {
                break;
            }

            last_write_time = cost_time;
        }

        // 正向传播，存储各层的神经元函数输入及神经元输出
        network->AssignOutput(n_vector_vector_matrix,
                               a_vector_vector_matrix,
                               a_end_vector_matrix);
        LOGTRACE("w_vector_matrix = " << network->GetWVectorMatrix() <<
                 ", b_vector_matrix = " << network->GetBVectorMatrix() <<
                 ", p_vector_matrix = " << p_vector_matrix << 
                 ", n_vector_vector_matrix = " << n_vector_vector_matrix << 
                 ", a_vector_vector_matrix = " << a_vector_vector_matrix);

        // 是否满足停止条件
        training_information.SetError(error_function->GetError(a_end_vector_matrix,
                                                               t_vector_matrix,
                                                               a_minus_t_vector_matrix,
                                                               a_minus_t_vector_matrix_mid));
        LOGTRACE("training_information.GetError() = " << training_information.GetError());
        if (training_information.IsSatisfied(stop_condition)) {
            LOGDEBUG("conditions satisfied, break");
            break;
        }

        // w
        network->AssignWVectorMatrix(w_vector_matrix);

        // 反向传播
        // 最后一层的雅克比矩阵
        network->GetActiveFunction(layer_num - 1)->
            AssignDerivativeVectorMatrixMultiThread(n_vector_vector_matrix[layer_num - 1],
                                                    f_vector_vector_matrix[layer_num - 1]);
        // 最后一层的敏感性
        VectorMatrixMultiThreadCalculator::AssignDiagonaledTimes(
            f_vector_vector_matrix[layer_num - 1],
            VectorMatrixMultiThreadCalculator::AssignMinus(a_end_vector_matrix,
                                                           t_vector_matrix,
                                                           a_minus_t_vector_matrix),
            s_vector_vector_matrix[layer_num - 1]);

        // 从倒数第二层开始
        for (i_layer = layer_num - 1; i_layer > 0; -- i_layer) {
            // 获取本层神经元函数的雅克比矩阵
            network->GetActiveFunction(i_layer - 1)->AssignDerivativeVectorMatrixMultiThread(
                n_vector_vector_matrix[i_layer - 1],
                f_vector_vector_matrix[i_layer - 1]);
            // 计算本层敏感性
            VectorMatrixMultiThreadCalculator::AssignTimes(
                VectorMatrixMultiThreadCalculator::AssignDiagonaledTimesTransformed(
                    f_vector_vector_matrix[i_layer - 1],
                    w_vector_matrix[i_layer],
                    f_times_w_vector_vector_matrix[i_layer - 1]),
                s_vector_vector_matrix[i_layer],
                s_vector_vector_matrix[i_layer - 1]);
        }

        // 获取w，b的改变量
        LOGTRACE("s_vector_vector_matrix = " << s_vector_vector_matrix <<
                 ", a_vector_vector_matrix.T() = " << a_vector_vector_matrix.T());
        VectorMatrixMultiThreadCalculator::AssignTimesTransformed(s_vector_vector_matrix, 
                                                                  a_vector_vector_matrix, 
                                                                  delta_w_vector_vector_matrix);
        LOGTRACE("delta_w_vector_vector_matrix(should be minus) = " << delta_w_vector_vector_matrix <<
                 ", delta_b_vector_vector_matrix(should be minus) = " << s_vector_vector_matrix);

        // 获取w, b改变量的整体值
        error_function->AssignGlobalWeightMultiThread(delta_w_vector_vector_matrix,
                                                      delta_w_vector_matrix);
        error_function->AssignGlobalWeightMultiThread(s_vector_vector_matrix,
                                                      delta_b_vector_matrix);

        // 获取当前值
        network->AssignBVectorMatrix(b_vector_matrix);

        // 寻找共轭点
        learning_rate_a = 0;
        error_a = error_function->GetError(network->AssignOutput(n_vector_vector_matrix, 
                                                                 a_vector_vector_matrix, 
                                                                 a_end_vector_matrix,
                                                                 AssignPlus(AssignTimes(delta_w_vector_matrix, 
                                                                                        - learning_rate_a,
                                                                                        final_w_vector_matrix), 
                                                                            w_vector_matrix, 
                                                                            final_w_vector_matrix),
                                                                 AssignPlus(AssignTimes(delta_b_vector_matrix, 
                                                                                        - learning_rate_a,
                                                                                        final_b_vector_matrix), 
                                                                            b_vector_matrix, 
                                                                            final_b_vector_matrix)),
                                           t_vector_matrix, 
                                           a_minus_t_vector_matrix, 
                                           a_minus_t_vector_matrix_mid);
        error_c = error_function->GetError(network->AssignOutput(n_vector_vector_matrix, 
                                                                 a_vector_vector_matrix, 
                                                                 a_end_vector_matrix,
                                                                 AssignPlus(AssignTimes(delta_w_vector_matrix, 
                                                                                        - learning_rate_c,
                                                                                        final_w_vector_matrix), 
                                                                            w_vector_matrix, 
                                                                            final_w_vector_matrix),
                                                                 AssignPlus(AssignTimes(delta_b_vector_matrix, 
                                                                                        - learning_rate_c,
                                                                                        final_b_vector_matrix), 
                                                                            b_vector_matrix, 
                                                                            final_b_vector_matrix)),
                                           t_vector_matrix, 
                                           a_minus_t_vector_matrix, 
                                           a_minus_t_vector_matrix_mid);
        LOGDEBUG("error_a = " << error_a << 
                 ", error_c = " << error_c <<
                 ", learning_rate_a = " << learning_rate_a << 
                 ", learning_rate_c = " << learning_rate_c);

        if (error_a < error_c) { // . '
            do {
                learning_rate_b = learning_rate_c;
                error_b = error_c;
                learning_rate_c = learning_rate_a + (learning_rate_b - learning_rate_a) * (1 - MW_MATH_GOLDEN_RATIO);
                error_c = error_function->GetError(network->AssignOutput(n_vector_vector_matrix, 
                                                                         a_vector_vector_matrix, 
                                                                         a_end_vector_matrix,
                                                                         AssignPlus(AssignTimes(delta_w_vector_matrix, 
                                                                                                - learning_rate_c,
                                                                                                final_w_vector_matrix), 
                                                                                    w_vector_matrix, 
                                                                                    final_w_vector_matrix),
                                                                         AssignPlus(AssignTimes(delta_b_vector_matrix, 
                                                                                                - learning_rate_c,
                                                                                                final_b_vector_matrix), 
                                                                                    b_vector_matrix, 
                                                                                    final_b_vector_matrix)),
                                                   t_vector_matrix, 
                                                   a_minus_t_vector_matrix, 
                                                   a_minus_t_vector_matrix_mid);
            } while (error_a < error_c); // . '
            // ' . '
            LOGDEBUG("error_a = " << error_a << 
                     ", error_c = " << error_c <<
                     ", error_b = " << error_b << 
                     ", learning_rate_a = " << learning_rate_a <<
                     ", learning_rate_c = " << learning_rate_c << 
                     ", learning_rate_b = " << learning_rate_b);

            if (learning_rate_c == learning_rate_a) {
                LOGDEBUG("learning_rate_c == " << learning_rate_a <<
                         ", network reaches it's limit, break.");
                break;
            }
        } else { // ' .
            learning_rate_b = learning_rate_a + (learning_rate_c - learning_rate_a) / (1 - MW_MATH_GOLDEN_RATIO);
            error_b = error_function->GetError(network->AssignOutput(n_vector_vector_matrix, 
                                                                     a_vector_vector_matrix, 
                                                                     a_end_vector_matrix,
                                                                     AssignPlus(AssignTimes(delta_w_vector_matrix, 
                                                                                            - learning_rate_b,
                                                                                            final_w_vector_matrix), 
                                                                                w_vector_matrix, 
                                                                                final_w_vector_matrix),
                                                                     AssignPlus(AssignTimes(delta_b_vector_matrix, 
                                                                                            - learning_rate_b,
                                                                                            final_b_vector_matrix), 
                                                                                b_vector_matrix, 
                                                                                final_b_vector_matrix)),
                                                                     t_vector_matrix, 
                                                                     a_minus_t_vector_matrix, 
                                                                     a_minus_t_vector_matrix_mid);

            while (error_c > error_b) { // ' .
                learning_rate_a = learning_rate_c;
                error_a = error_c;
                learning_rate_c = learning_rate_b;
                error_c = error_b;

                learning_rate_b = learning_rate_a + (learning_rate_c - learning_rate_a) / (1 - MW_MATH_GOLDEN_RATIO);
                error_b = error_function->GetError(network->AssignOutput(n_vector_vector_matrix, 
                                                                         a_vector_vector_matrix, 
                                                                         a_end_vector_matrix,
                                                                         AssignPlus(AssignTimes(delta_w_vector_matrix, 
                                                                                                - learning_rate_b,
                                                                                                final_w_vector_matrix), 
                                                                                    w_vector_matrix, 
                                                                                    final_w_vector_matrix),
                                                                         AssignPlus(AssignTimes(delta_b_vector_matrix, 
                                                                                                - learning_rate_b,
                                                                                                final_b_vector_matrix), 
                                                                                    b_vector_matrix, 
                                                                                    final_b_vector_matrix)),
                                                   t_vector_matrix, 
                                                   a_minus_t_vector_matrix, 
                                                   a_minus_t_vector_matrix_mid);
            }
            // ' . '
            LOGDEBUG("error_a = " << error_a << 
                     ", error_c = " << error_c <<
                     ", error_b = " << error_b << 
                     ", learning_rate_a = " << learning_rate_a <<
                     ", learning_rate_c = " << learning_rate_c << 
                     ", learning_rate_b = " << learning_rate_b);
        }

        learning_rate_d = learning_rate_a + (learning_rate_b - learning_rate_a) * MW_MATH_GOLDEN_RATIO;
        error_d = error_function->GetError(network->AssignOutput(n_vector_vector_matrix, 
                                                                 a_vector_vector_matrix, 
                                                                 a_end_vector_matrix,
                                                                 AssignPlus(AssignTimes(delta_w_vector_matrix, 
                                                                                        - learning_rate_d,
                                                                                        final_w_vector_matrix), 
                                                                            w_vector_matrix, 
                                                                            final_w_vector_matrix),
                                                                 AssignPlus(AssignTimes(delta_b_vector_matrix, 
                                                                                        - learning_rate_d,
                                                                                        final_b_vector_matrix), 
                                                                            b_vector_matrix, 
                                                                            final_b_vector_matrix)),
                                           t_vector_matrix, 
                                           a_minus_t_vector_matrix, 
                                           a_minus_t_vector_matrix_mid);
        LOGDEBUG("error_a = " << error_a << 
                           ", error_c = " << error_c <<
                           ", error_d = " << error_d << 
                           ", error_b = " << error_b << 
                           ", learning_rate_a = " << learning_rate_a << 
                           ", learning_rate_c = " << learning_rate_c <<
                           ", learning_rate_d = " << learning_rate_d << 
                           ", learning_rate_b = " << learning_rate_b);
        iteration_count = 0;

        // a 39 c 22 d 39 b
        while (error_c != error_d && learning_rate_c < learning_rate_d) {
            ++ iteration_count;

            if (error_c < error_d) {
                learning_rate_b = learning_rate_d;
                error_b = error_d;
                learning_rate_d = learning_rate_c;
                error_d = error_c;
                learning_rate_c = learning_rate_a + (learning_rate_b - learning_rate_a) *
                                                    (1 - MW_MATH_GOLDEN_RATIO);
                error_c = error_function->GetError(network->AssignOutput(n_vector_vector_matrix, 
                                                                         a_vector_vector_matrix, 
                                                                         a_end_vector_matrix,
                                                                         AssignPlus(AssignTimes(delta_w_vector_matrix, 
                                                                                                - learning_rate_c,
                                                                                                final_w_vector_matrix), 
                                                                                    w_vector_matrix, 
                                                                                    final_w_vector_matrix),
                                                                         AssignPlus(AssignTimes(delta_b_vector_matrix, 
                                                                                                 - learning_rate_c,
                                                                                                 final_b_vector_matrix), 
                                                                                    b_vector_matrix, 
                                                                                    final_b_vector_matrix)),
                                                   t_vector_matrix, 
                                                   a_minus_t_vector_matrix, 
                                                   a_minus_t_vector_matrix_mid);
            } else {
                learning_rate_a = learning_rate_c;
                error_a = error_c;
                learning_rate_c = learning_rate_d;
                error_c = error_d;
                learning_rate_d = learning_rate_a + (learning_rate_b - learning_rate_a) * MW_MATH_GOLDEN_RATIO;
                error_d = error_function->GetError(network->AssignOutput(n_vector_vector_matrix, 
                                                                         a_vector_vector_matrix, 
                                                                         a_end_vector_matrix,
                                                                         AssignPlus(AssignTimes(delta_w_vector_matrix, 
                                                                                                - learning_rate_d,
                                                                                                final_w_vector_matrix), 
                                                                                    w_vector_matrix, 
                                                                                    final_w_vector_matrix),
                                                                         AssignPlus(AssignTimes(delta_b_vector_matrix, 
                                                                                                - learning_rate_d,
                                                                                                final_b_vector_matrix), 
                                                                                    b_vector_matrix, 
                                                                                    final_b_vector_matrix)),
                                                   t_vector_matrix, 
                                                   a_minus_t_vector_matrix, 
                                                   a_minus_t_vector_matrix_mid);
            }
            LOGDEBUG("error_a = " << error_a << 
                     ", error_c = " << error_c <<
                     ", error_d = " << error_d << 
                     ", error_b = " << error_b << 
                     ", learning_rate_a = " << learning_rate_a << 
                     ", learning_rate_c = " << learning_rate_c <<
                     ", learning_rate_d = " << learning_rate_d << 
                     ", learning_rate_b = " << learning_rate_b);
        }

        LOGTRACE("delta_w_vector_matrix = " << delta_w_vector_matrix <<
                 ", delta_b_vector_matrix = " << delta_b_vector_matrix << 
                 ", learning_rate = " << learning_rate_c);
        // 更新W, b
        network->SetWVectorMatrix(AssignPlus(w_vector_matrix,
                                             AssignTimes(delta_w_vector_matrix, - learning_rate_c, delta_w_vector_matrix),
                                             w_vector_matrix));
        network->SetBVectorMatrix(AssignPlus(b_vector_matrix,
                                             AssignTimes(delta_b_vector_matrix, - learning_rate_c, delta_b_vector_matrix),
                                             b_vector_matrix));

        training_information.Age();

        cost_time = training_information.GetCostTime();
        LOGINFO("epoch = " << training_information.GetEpoch() << 
                ", error = " << error_c << 
                ", learning_rate_c = " << learning_rate_c <<
                ", iteration_count = " << iteration_count << 
                ", cost_time = " << cost_time <<
                ", left_time = " << (stop_condition->GetTimeLimit() - cost_time));

        if (iteration_count == 0) {
            LOGDEBUG("iteration_count == 0, break");
            break;
        }

    }

    if (network_file_name_prefix != NULL) {
        SaveNetwork(MWString((*network_file_name_prefix) + 
                              MWString::ToString(training_information.GetEpoch()) + "-" + 
                              MWString::ToString(error_c)),
                    network);
    }

    return training_information;
}

int BackPropgation::SaveNetwork(const MWString &network_file_name,
                                NeuralNetwork *network) const {
    BinFile network_file(network_file_name);
    if (!network_file.OpenForWrite()) {
        LOGERROR("open file " << network_file_name << " for write error");
        return -1;
    }
    if (!network_file.Write(network->Serialize())) {
        LOGERROR("write char_vector to " << network_file_name << " error");
        network_file.Close();
        return -1;
    }
    network_file.Close();
    LOGINFO("network saved to " << network_file_name);
    return 0;
}
