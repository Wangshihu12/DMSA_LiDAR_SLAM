/* Copyright (C) 2024 David Skuddis - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: davidskuddis@web.de, or visit: https://opensource.org/license/mit/
 */

#ifndef MAPMANAGEMENT_H
#define MAPMANAGEMENT_H

#include <eigen3/Eigen/Core>
#include "OptimizablePointSet.h"
#include "ConsecutivePoses.h"
#include "KeyframeData.h"

using namespace pcl;
using namespace Eigen;

class MapManagement : public OptimizablePointSet<PointNormal>
{
public:
    bool isInitialized = false;         // 地图是否被初始化

    StampedConsecutivePoses keyframePoses;          // 关键帧位姿信息

    RingBuffer<KeyframeData> keyframeDataBuffer;    // 关键帧信息的环形缓冲区

    PointCloud<PointStampId>::Ptr activePoints;     // 下采样后的静态点

    vector<int> keyframeIds;
    vector<int> ringIds;        // 根据全局点的 ID 获取环形缓冲区的 ID

    int maxNumKeyframes;    // 最大关键帧数量

    // new origin for centralization
    Vector3d origin;    // 地图原点

    bool useGravityErrorTerms = false;      // 是否使用重力误差项
    bool useOdometryErrorTerms = false;     // 是否使用里程计误差项
    VectorXd gravityErrorTerm;      // 存储重力误差项
    Vector3d gravity;   // 重力方向

    VectorXd odometryErrorTerm;     // 里程计误差项

    VectorXd additionalErrors;      // 其他额外的误差项

    double std_dev_acc = 0.3;       // 加速度的标准差
    Matrix3d odometryTranslCovInv;  // 里程计平移误差的逆协方差矩阵
    Matrix3d odometryOrientCovInv;  // 里程计方向误差的逆协方差矩阵

    Eigen::Matrix3d Cov_grav_inv;           // 重力误差的逆协方差矩阵
    double balancingFactorGrav = 1.0;       // 重力误差项的平衡因子
    double balancingFactorOdom = 1000.0;    // 里程计误差项的平衡因子

    MapManagement(int n_max = 30) : keyframePoses(n_max)
    {
        maxNumKeyframes = n_max;

        keyframeDataBuffer.init(n_max);

        activePoints = PointCloud<PointStampId>::Ptr(new PointCloud<PointStampId>());

        gravity << 0.0, 0.0, -9.805;

        Matrix3d Cov_grav = std::pow(std_dev_acc, 2) * Matrix3d::Identity();
        Cov_grav_inv = Cov_grav.inverse();

        odometryTranslCovInv = (std::pow(0.01, 2) * Matrix3d::Identity()).inverse();
        odometryOrientCovInv = (std::pow(0.01, 2) * Matrix3d::Identity()).inverse();
    }

    // 重新设置地图原点，中心化
    void centralize()
    {
        return;
        origin = keyframePoses.relativePoses.Translations.col(0);
        keyframePoses.relativePoses.Translations.col(0).setZero();
        keyframePoses.relative2global();
    }

    // 去中心化，将地图移动回原始原点
    void decentralize()
    {
        return;
        keyframePoses.global2relative();
        keyframePoses.relativePoses.Translations.col(0) = origin;
        keyframePoses.relative2global();
    }

    // 更新所有点的全局坐标
    void updateGlobalPoints()
    {
        int globalId = 0;
        Matrix4f currTransform = Matrix4f::Identity();
        Matrix3f currRot;

        this->minGridSize = std::numeric_limits<float>::max();

        for (int k = 0; k < keyframeDataBuffer.getNumElements(); ++k)
        {
            // update grid size
            this->minGridSize = std::min(this->minGridSize, keyframeDataBuffer.at(k).gridSize);
            // 当前关键帧的点云
            PointCloud<PointNormal> &currCloud = *keyframeDataBuffer.at(k).pointCloudLocal;
            // 计算当前关键帧的变换矩阵
            currRot = axang2rotm(keyframePoses.globalPoses.Orientations.col(k)).cast<float>();

            currTransform.block(0, 0, 3, 3) = currRot;
            currTransform.block(0, 3, 3, 1) = (keyframePoses.globalPoses.Translations.col(k)).cast<float>();

            // 更新全局点
            for (auto &point : currCloud)
            {
                globalPoints.points[globalId].getVector4fMap() = currTransform * point.getVector4fMap();

                globalPoints.points[globalId].getNormalVector3fMap() = currRot * point.getNormalVector3fMap();

                ++globalId;
            }
        }
    }

    // 返回误差项
    Eigen::VectorXd &getAdditionalErrorTerms()
    {
        if (useGravityErrorTerms == true && useOdometryErrorTerms == false)
            return gravityErrorTerm;

        if (useGravityErrorTerms == false && useOdometryErrorTerms == true)
            return odometryErrorTerm;

        return additionalErrors;
    }

    // 更新误差项
    int updateAdditionalErrors()
    {
        if (useGravityErrorTerms == false && useOdometryErrorTerms == false)
            return 0;

        if (useGravityErrorTerms == true && useOdometryErrorTerms == false)
        {
            this->updateGravityErrors();

            return gravityErrorTerm.size();
        }

        if (useGravityErrorTerms == false && useOdometryErrorTerms == true)
        {
            this->updateOdometryErrors();

            return odometryErrorTerm.size();
        }

        this->updateGravityErrors();

        this->updateOdometryErrors();

        additionalErrors.conservativeResize(gravityErrorTerm.size() + odometryErrorTerm.size());

        additionalErrors << gravityErrorTerm, odometryErrorTerm;

        return additionalErrors.size();
    }

    // 获取关键帧的相对位姿参数
    void getPoseParameters(Eigen::VectorXd &params)
    {
        keyframePoses.relativePoses.getParamsAsVector(params);
    }

    // 设置关键帧的位姿参数
    void setPoseParameters(const Eigen::VectorXd &params)
    {
        keyframePoses.relativePoses.setParamsFromVector(params);

        keyframePoses.relative2global();
    }

    // 根据全局点的索引获取在哪个环形缓冲区的 ID
    const int &getIdOfPoint(int &globalPointIndex)
    {
        // return keyframeIds[globalPointIndex];
        return ringIds[globalPointIndex];
    }

    // 更新重力误差项
    void updateGravityErrors()
    {
        // resize
        gravityErrorTerm.conservativeResize(keyframeDataBuffer.getNumElements());

        // reset
        gravityErrorTerm.setZero();

        Eigen::Vector3d diffVec;

        for (int k = 1; k < keyframeDataBuffer.getNumElements(); ++k)
        {
            if (keyframeDataBuffer.at(k).gravityPlausible == false ) continue;

            // calculate gravity error
            diffVec = axang2rotm(keyframePoses.globalPoses.Orientations.col(k)) * keyframeDataBuffer.at(k).measuredGravity - gravity;

            // save
            gravityErrorTerm(k) = diffVec.transpose() * Cov_grav_inv * diffVec;
            gravityErrorTerm(k) *= balancingFactorGrav;
        }
    }

    void updateOdometryErrors()
    {
        odometryErrorTerm.conservativeResize(keyframeDataBuffer.getNumElements() - 1);
        odometryErrorTerm.setZero();

        Vector3d translDiff, orientDiff;
        // 从第二个关键帧开始
        for (int k = 1; k < keyframeDataBuffer.getNumElements(); ++k)
        {
            translDiff = keyframeDataBuffer.at(k).relativeTransl - keyframePoses.relativePoses.Translations.col(k);

            orientDiff = rotm2axang(axang2rotm(keyframePoses.relativePoses.Orientations.col(k)).transpose() * keyframeDataBuffer.at(k).relativeOrientMat);

            odometryErrorTerm(k - 1) += translDiff.transpose() * odometryTranslCovInv * translDiff;
            odometryErrorTerm(k - 1) += orientDiff.transpose() * odometryOrientCovInv * orientDiff;

            odometryErrorTerm(k - 1) *= balancingFactorOdom;
        }
    }

    // 根据给定的关键帧范围创建一个子地图
    std::shared_ptr<MapManagement> getSubmap(int fromId, int toId)
    {
        int nFrames = toId - fromId + 1;

        std::shared_ptr<MapManagement> subset = std::shared_ptr<MapManagement>(new MapManagement(nFrames));

        subset->minGridSize = std::numeric_limits<float>::max();

        for (int k = fromId; k <= toId; ++k)
        {
            Ref<Vector3d> currTransl = this->keyframePoses.globalPoses.Translations.col(k);
            Ref<Vector3d> currOrient = this->keyframePoses.globalPoses.Orientations.col(k);
            double &currStamp = this->keyframePoses.stamps(k);

            subset->addKeyframe(currTransl, currOrient, currStamp, this->keyframeDataBuffer.at(k));

            subset->minGridSize = std::min(subset->minGridSize, this->keyframeDataBuffer.at(k).gridSize);
        }

        subset->keyframePoses.global2relative();

        return subset;
    }

    // 从子地图中更新当前地图的关键帧信息
    void updatePosesFromSubmap(int fromId, int toId, MapManagement &submap)
    {
        submap.keyframePoses.global2relative();

        int numParams = toId - fromId + 1;

        this->keyframePoses.relativePoses.Translations.block(0, fromId + 1, 3, numParams - 1) = submap.keyframePoses.relativePoses.Translations.block(0, 1, 3, numParams - 1);
        this->keyframePoses.relativePoses.Orientations.block(0, fromId + 1, 3, numParams - 1) = submap.keyframePoses.relativePoses.Orientations.block(0, 1, 3, numParams - 1);

        this->keyframePoses.relative2global();
    }

    // 将关键帧的局部点云转换到全局坐标系
    void getGlobalKeyframeCloud(int keyId, PointCloud<PointNormal>::Ptr pcTarget)
    {

        Matrix4f transformToGlobal = Matrix4f::Identity();

        transformToGlobal.block(0, 0, 3, 3) = axang2rotm(keyframePoses.globalPoses.Orientations.col(keyId)).cast<float>();
        transformToGlobal.block(0, 3, 3, 1) = (keyframePoses.globalPoses.Translations.col(keyId)).cast<float>();

        transformPointCloudWithNormals(*keyframeDataBuffer.at(keyId).pointCloudLocal, *pcTarget, transformToGlobal);
    }

    // 计算环形缓冲区中所有关键帧局部点云点的总数
    int getNumPointsInBuffer()
    {
        int numPts = 0;

        for (int k = 0; k < keyframeDataBuffer.getNumElements(); ++k)
            numPts += keyframeDataBuffer.at(k).pointCloudLocal->size();

        return numPts;
    }

    // 向环形缓冲区中添加新的关键帧
    void addKeyframe(Ref<Vector3d> position_w, Ref<Vector3d> orient_w, double stamp, KeyframeData keyframeData)
    {
        keyframePoses.relative2global();

        if (keyframeDataBuffer.isFull() == false)
        {
            int currId = keyframeDataBuffer.getNumElements();

            keyframePoses.globalPoses.Translations.col(currId) = position_w;
            keyframePoses.globalPoses.Orientations.col(currId) = orient_w;
            keyframePoses.stamps(currId) = stamp;
        }
        else
        {
            // shift 将第一列后面的元素赋值给左边，相当于全部左移
            keyframePoses.globalPoses.Translations.block(0, 0, 3, maxNumKeyframes - 1) = keyframePoses.globalPoses.Translations.block(0, 1, 3, maxNumKeyframes - 1);
            keyframePoses.globalPoses.Orientations.block(0, 0, 3, maxNumKeyframes - 1) = keyframePoses.globalPoses.Orientations.block(0, 1, 3, maxNumKeyframes - 1);

            keyframePoses.stamps.segment(0, maxNumKeyframes - 1) = keyframePoses.stamps.segment(1, maxNumKeyframes - 1);

            // add new 然后把新数据放在最后一列
            keyframePoses.globalPoses.Translations.col(maxNumKeyframes - 1) = position_w;
            keyframePoses.globalPoses.Orientations.col(maxNumKeyframes - 1) = orient_w;
            keyframePoses.stamps(maxNumKeyframes - 1) = stamp;
        }
        // 转换为相对位姿
        keyframePoses.global2relative();

        // save odometry result 将相对位姿保存到 keyframeData
        if (keyframeDataBuffer.isFull() == false)
        {
            int currId = keyframeDataBuffer.getNumElements();

            keyframeData.relativeTransl = keyframePoses.relativePoses.Translations.col(currId);
            keyframeData.relativeOrient = keyframePoses.relativePoses.Orientations.col(currId);

            keyframeData.relativeOrientMat = axang2rotm(keyframeData.relativeOrient);
        }
        else
        {
            keyframeData.relativeTransl = keyframePoses.relativePoses.Translations.col(maxNumKeyframes - 1);
            keyframeData.relativeOrient = keyframePoses.relativePoses.Orientations.col(maxNumKeyframes - 1);

            keyframeData.relativeOrientMat = axang2rotm(keyframeData.relativeOrient);
        }

        // set padding variable to 1
        for (auto & point : keyframeData.pointCloudLocal->points) point.data[3] = 1.0f;

        // add keyframe data to ringbuffer
        keyframeDataBuffer.addElem(keyframeData);

        if (isInitialized == false)
            isInitialized = true;

        // update global points
        int numGlobalPts = getNumPointsInBuffer();

        globalPoints.resize(numGlobalPts);
        keyframeIds.resize(numGlobalPts);
        ringIds.resize(numGlobalPts);

        // update global map
        updateGlobalPoints();

        // update keyframe ids
        int globalId = 0;

        for (int k = 0; k < keyframeDataBuffer.getNumElements(); ++k)
        {
            for (int j = 0; j < static_cast<int>(keyframeDataBuffer.at(k).pointCloudLocal->points.size()); ++j)
            {
                keyframeIds[globalId] = k;
                ringIds[globalId] = keyframeDataBuffer.at(k).ringIds(j);

                ++globalId;
            }
        }
    }
};

#endif
