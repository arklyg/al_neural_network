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


#define PROGRAMM_NAME "testreloadneuralnetwork"

int main(int argv, const char *args[]) {
    if (argv < 2) {
        std::cout << "usage: " << args[0] << " network_file_name" << std::endl;
        return EXIT_FAILURE;
    }

    InitLogger(PROGRAMM_NAME);

    MWString network_file_name(args[1]);
    BinFile network_file(network_file_name);
    if (!network_file.OpenForRead()) {
        LOGERROR("open " << network_file_name << " for read error");
        return EXIT_FAILURE;
    }

    std::vector<char> char_vector = network_file.Read();
    if (char_vector.size() == 0) {
        LOGERROR("read 0");
        return EXIT_FAILURE;
    }
    network_file.Close();

    NeuralNetwork network;
    network.Deserialize(char_vector, 0);
    network.InitializeActiveFunction();

    LOGINFO("network.GetWVectorMatrix() = " <<
             network.GetWVectorMatrix() << ", network.GetBVectorMatrix() = " <<
             network.GetBVectorMatrix() << ", network.GetNeuronNumVector() = " <<
             network.GetNeuronNumVector());

    Vector<Vector<Matrix<Data> > > n_vector_vector_matrix, 
                                   a_vector_vector_matrix;
    Vector<Matrix<Data> > a_end_vector_matrix;
    network.AssignOutputStructure(1, n_vector_vector_matrix,
                                     a_vector_vector_matrix, a_end_vector_matrix);
    a_vector_vector_matrix[0] = Vector<Matrix<Data> >(1, Matrix<Data>(2, 1));

    VectorMatrixMultiThreadCalculator::Initialize(1);
    MWMathFunction::Initialize(1);
    while (true) {
        std::cout << "input first double x, (1 < x < 2): ";
        std::cin >> a_vector_vector_matrix[0][0]._data[0][0];
        std::cout << "input second double y, (1 < y < 2): ";
        std::cin >> a_vector_vector_matrix[0][0]._data[1][0];
        network.AssignOutput(n_vector_vector_matrix, a_vector_vector_matrix,
                             a_end_vector_matrix);
        std::cout << "x * y, network result = " << a_end_vector_matrix[0]._data[0][0] <<
                     ", result should be " << (a_vector_vector_matrix[0][0]._data[0][0] * 
                        a_vector_vector_matrix[0][0]._data[1][0]) << 
                     ", error = " << (pow(a_end_vector_matrix[0]._data[0][0] - 
                        a_vector_vector_matrix[0][0]._data[0][0] * a_vector_vector_matrix[0][0]._data[1][0], 2)) << 
                     std::endl;
    }

    return EXIT_SUCCESS;
}
