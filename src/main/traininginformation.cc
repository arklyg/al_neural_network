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


#include "traininginformation.h"

TrainingInformation::TrainingInformation(Data error, size_t epoch, double time)
    : _error(error)
    , _epoch(epoch)
    , _time(time) {
}

bool TrainingInformation::IsSatisfied(const TrainingInformation *information)
    const {
    if (_error < information->GetError()) {
        LOGINFO("(_error = " << _error << 
                ") < (tarGetError = " << information->GetError() << 
                "), satisfied");
        return true;
    }
    // epoch 为零则不作为满足条件
    if (information->GetEpoch() > 0 && _epoch >= information->GetEpoch()) {
        LOGINFO("(_epoch = " << _epoch << 
                ") >= (tarGetEpoch = " << information->GetEpoch() << 
                "), satisfied");
        return true;
    }
    // time_limit 为零则不作为满足条件
    if (information->GetTimeLimit() > 0
            && this->GetCostTime() > information->GetTimeLimit()) {
        LOGINFO("(cost_time = " << this->GetCostTime() <<
                ") > (_time_limit = " << information->GetTimeLimit() << 
                "), satisfied");
        return true;
    }

    return false;
}
