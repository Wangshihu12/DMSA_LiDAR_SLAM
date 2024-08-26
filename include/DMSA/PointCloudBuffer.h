/* Copyright (C) 2024 David Skuddis - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: davidskuddis@web.de, or visit: https://opensource.org/license/mit/
 */

#ifndef POINTCLOUDBUFFER_H
#define POINTCLOUDBUFFER_H

#include "PointCloudPlus.h"
#include <vector>
#include <limits>
#include <iostream>
#include "RingBuffer.h"

using namespace pcl;
using namespace std;

class PointCloudBuffer : public RingBuffer<PointCloudPlus>
{
public:
    // 返回点云数据中最大和最小的时间戳
    void getMinMaxPointStamps(double &minStamp, double &maxStamp)
    {
        minStamp = std::numeric_limits<double>::max();
        maxStamp = 0.0;

        for (int k = 0; k < std::min(numUpdates, maxElements); ++k)
        {
            for (const auto & point : data[k].points)
            {
                if (point.stamp < minStamp)
                    minStamp = point.stamp;
                if (point.stamp > maxStamp)
                    maxStamp = point.stamp;
            }
        }
    }

    // 计算储存在环形缓冲区的点的总数，每个元素是一帧点云
    int getNumPoints()
    {
        int numPts = 0;

        for (int k = 0; k < getNumElements(); ++k)
            numPts += this->at(static_cast<int>(k)).size();

        return numPts;
    }
};

#endif
