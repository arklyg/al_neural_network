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


#ifndef _TRAINING_INFORMATION_
#define _TRAINING_INFORMATION_

#include <float.h>

#include <mwtimer.h>
#include <mwmatrix.h>

#include "neuralnetworkglobal.h"

#define TRAINING_INFORMATION_DEFAULT_ERROR 0.01

class TrainingInformation {
  private:
    Data _error;
    size_t _epoch;
    double _time; // 做info时是start_time 做condition时是time_limit

  public:
    TrainingInformation(Data error = TRAINING_INFORMATION_DEFAULT_ERROR,
                        size_t epoch = 0, double time = 0);

    inline void SetError(Data error);
    inline void SetEpoch(size_t epoch);
    inline void SetTimeLimit(double time_limit);
    inline void SetStartTime(double start_time);
    inline void SetStartTime();

    inline Data GetError() const;
    inline size_t GetEpoch() const;
    inline double GetTimeLimit() const;
    inline double GetStartTime() const;
    inline double GetCostTime() const;

    inline size_t Age();

    bool IsSatisfied(const TrainingInformation *information) const;
};

inline void TrainingInformation::SetError(Data error) {
    _error = error;
}

inline void TrainingInformation::SetEpoch(size_t epoch) {
    _epoch = epoch;
}

inline void TrainingInformation::SetTimeLimit(double time_limit) {
    _time = time_limit;
}

inline void TrainingInformation::SetStartTime(double start_time) {
    _time = start_time;
}

inline void TrainingInformation::SetStartTime() {
    return SetStartTime(MWTimer::GetTime());
}

inline Data TrainingInformation::GetError() const {
    return _error;
}

inline size_t TrainingInformation::GetEpoch() const {
    return _epoch;
}

inline double TrainingInformation::GetTimeLimit() const {
    return _time;
}

inline double TrainingInformation::GetStartTime() const {
    return _time;
}

inline size_t TrainingInformation::Age() {
    return ++ _epoch;
}

inline double TrainingInformation::GetCostTime() const {
    return MWTimer::Endure(_time);
}

#endif
