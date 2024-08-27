/* Copyright (C) 2024 David Skuddis - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: davidskuddis@web.de, or visit: https://opensource.org/license/mit/
 */

#ifndef CONTINUOUSTRAJECTORY_H
#define CONTINUOUSTRAJECTORY_H

#include "OptimizablePointSet.h"
#include "ConsecutivePoses.h"
#include <vector>
#include "PointCloudBuffer.h"
#include "ImuBuffer.h"
#include "ImuPreintegration.h"
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/interpolators/cubic_b_spline.hpp>

using namespace Eigen;
using namespace std;

class ContinuousTrajectory : public OptimizablePointSet<PointStampId>
{
public:
    StampedConsecutivePoses controlPoses;   // 有时间戳的连续控制姿态

    Poses denseGlobalPoses;     // 全局点的姿态

    vector<Matrix4f> denseTformsLocal2Global;   // local 2 global 转换矩阵

    VectorXd trajTime;  // 轨迹时间点
    Vector3d gravity;   // 重力向量

    // preintegrated imu measurements
    vector<Matrix3d> preintImuRots;         // 预积分 IMU 的旋转矩阵
    vector<Vector3d> preintRelPositions;    // 预积分相对位置
    vector<Vector3d> preintRelVelocity;     // 预积分相对速度

    Vector3d preintPosComplHor;     // 水平方向的预积分位置补偿

    vector<Matrix<double, 9, 9>> CovPVRot_inv;  // 位置、速度和旋转的逆协方差矩阵

    double t0;      // 轨迹的起始时间
    double horizon; // 轨迹的预测时间范围

    double dt_res = 0.0001;     // 时间分辨率

    bool validImuData = false;  // IMU 数据是否有效

    double balancingImu = 0.001f;   // 平衡IMU数据的影响
    bool useImuErrorTerms;          // 是否使用IMU误差项

    int numParams;  // 参数的数量
    int n_total;    // 参数的总数

    // origin is saved here in case of centralization
    Vector3d origin;    // 中心化后的原点位置

    // imu measurement containers
    Matrix3Xd accMeas;      // 加速度测量数据
    Matrix3Xd angVelMeas;   // 角速度测量数据

    VectorXd imuFactorError;    // IMU 误差项
    VectorXi paramIndices;      // 参数的索引

    // registered point cloud buffer
    std::shared_ptr<PointCloudBuffer> regPcBuffer;  // 点云环形缓冲区

    vector<vector<int>> tformIdPerPoint;    // 存储了每个点对应的变换ID列表

    ContinuousTrajectory() {}

    // 将轨迹的中心化，即将轨迹的起始点移动到原点
    void centralize()
    {
        // 保存原始起点
        origin = controlPoses.relativePoses.Translations.col(0);
        // 将轨迹起点设置为 0
        controlPoses.relativePoses.Translations.col(0).setZero();
        // 根据新的相对姿态计算全局姿态
        controlPoses.relative2global();

        // 中心化静态点
        for (auto &point : globalPoints)
        {
            if (point.isStatic > 0)
                point.getVector3fMap() = point.getVector3fMap() - origin.cast<float>();
        }
    }

    // 去中心化，将中心化后的轨迹恢复到原来的位置
    void decentralize()
    {
        controlPoses.global2relative();
        controlPoses.relativePoses.Translations.col(0) = origin;
        controlPoses.relative2global();

        // 去中心化静态点
        for (auto &point : globalPoints)
        {
            if (point.isStatic > 0)
                point.getVector3fMap() = point.getVector3fMap() + origin.cast<float>();
        }
    }

    // 返回 IMU 误差因子
    Eigen::VectorXd &getAdditionalErrorTerms()
    {
        return imuFactorError;
    }

    // 更新 IMU 误差因子
    int updateAdditionalErrors()
    {
        if (useImuErrorTerms)
        {
            updateImuError();

            return imuFactorError.size();
        }
        else
            return 0;
    }

    // 将输入参数填充到控制点姿态中
    void getPoseParameters(VectorXd &params)
    {
        controlPoses.relativePoses.getParamsAsVector(params);
    }

    void setPoseParameters(const Eigen::VectorXd &params)
    {
        controlPoses.relativePoses.setParamsFromVector(params);
    }

    // 更新全局点云，将局部点云转换到全局坐标系
    void updateGlobalPoints()
    {
        // update dense transforms to global frame
        // 更新全局变换矩阵
        updateTrajDenseTforms();

        // transform points to global cloud
        int currId = 0;
        // 遍历缓冲区的所有点云，将每个点从局部坐标系转换到全局坐标系
        for (int pcId = 0; pcId < regPcBuffer->getNumElements(); ++pcId)
        {
            PointCloudPlus &currCloud = regPcBuffer->at(pcId);

            for (int k = 0; k < currCloud.size(); ++k)
            {
                PointStampId &pointOriginRef = currCloud.points[k];

                PointStampId &pointTargetRef = globalPoints.points[currId];

                int &currTformId = tformIdPerPoint[pcId][k];

                Eigen::Map<Vector4f> targetMap(pointTargetRef.data);

                targetMap = denseTformsLocal2Global[currTformId] * pointOriginRef.getVector4fMap();

                ++currId;
            }
        }
    }

    // 将一个静态点云添加到全局点云中，并标记这些点为静态
    void addStaticPoints(const pcl::PointCloud<PointStampId> &pcStatic)
    {
        int currSz = globalPoints.points.size();

        globalPoints.points.resize(currSz + static_cast<int>(pcStatic.size()));

        for (int k = currSz; k < globalPoints.points.size(); ++k)
        {
            globalPoints.points[k].getVector3fMap() = pcStatic.points[k - currSz].getVector3fMap();

            globalPoints.points[k].stamp = -1000.0;
            globalPoints.points[k].id = pcStatic.points[k - currSz].id;
            globalPoints.points[k].isStatic = 1;
        }
    }

    // 从全局点云中移除所有标记为静态的点
    void removeStaticPoints()
    {
        if (globalPoints.points.size() == 0)
            return;

        for (int k = 0; k < globalPoints.points.size(); ++k)
        {
            if (globalPoints.points[k].isStatic > 0)
            {
                globalPoints.points.resize(k);
                return;
            }
        }
    }

    // 更新从局部到全局的密集变换矩阵
    void updateTrajDenseTforms()
    {
        // 将局部姿态转换到全局姿态
        controlPoses.relative2global();

        // interpolate orientations with slerp
        for (int k = 0; k < n_total; ++k)
        {
            // rotational interpolation
            // 使用球面线性插值（slerp）来估计当前时间点的旋转
            getInterpRotation(controlPoses.globalPoses, controlPoses.stamps, trajTime(k), denseGlobalPoses.Orientations.col(k));
        }

        // interpolation of translation
        // 对每个轴，使用三次插值来估计平移
        for (int k = 0; k < 3; ++k)
        {

            // Map translations to std vec
            Eigen::VectorXd translations = controlPoses.globalPoses.Translations.row(k);
            Map<VectorXd> translationsMap(translations.data(), translations.size());
            std::vector<double> vec_translations(translationsMap.data(), translationsMap.data() + translationsMap.size());

            // Map XYZ to std vec
            Map<VectorXd> paramStampsMap(controlPoses.stamps.data(), controlPoses.stamps.size());
            std::vector<double> vec_paramStamps(paramStampsMap.data(), paramStampsMap.data() + paramStampsMap.size());

            // interpolation
            boost::math::barycentric_rational<double> s(vec_paramStamps.data(), vec_translations.data(), vec_paramStamps.size(), 2);

            for (int j = 0; j < n_total; ++j)
                denseGlobalPoses.Translations(k, j) = (double)s(trajTime(j));
        }

        // update dense transforms
        // 更新变换矩阵
        for (int k = 0; k < n_total; ++k)
        {
            denseTformsLocal2Global[k].block(0, 0, 3, 3) = axang2rotm(denseGlobalPoses.Orientations.col(k)).cast<float>();
            denseTformsLocal2Global[k].block(0, 3, 3, 1) = denseGlobalPoses.Translations.col(k).cast<float>();
        }
    }

    // 将一个点云缓冲区注册到全局点云中，并计算每个点对应的变换索引
    void registerPcBuffer(std::shared_ptr<PointCloudBuffer> &bufferPtrIn)
    {
        regPcBuffer = bufferPtrIn;

        globalPoints.resize(regPcBuffer->getNumPoints());

        // save minimum grid size for optimization
        // 计算最小网格尺寸
        this->minGridSize = std::numeric_limits<float>::max();

        for (int k = 0; k < regPcBuffer->getNumElements(); ++k)
            this->minGridSize = std::min(this->minGridSize, regPcBuffer->at(k).gridSize);

        // precalculate corresponding tform index for each point
        tformIdPerPoint.resize(regPcBuffer->getNumElements());

        int globalId = 0;

        for (int pcId = 0; pcId < regPcBuffer->getNumElements(); ++pcId)
        {
            PointCloudPlus &currCloud = regPcBuffer->at(pcId);

            tformIdPerPoint[pcId].resize(currCloud.size());

            for (int k = 0; k < currCloud.size(); ++k)
            {
                // 查找时间戳中与当前点时间戳最接近的值的指针
                auto pointerToVal = std::lower_bound(this->trajTime.data(), this->trajTime.data() + this->trajTime.size(), currCloud.points[k].stamp - this->t0);
                // 计算变换索引
                tformIdPerPoint[pcId][k] = std::min((int)std::distance(this->trajTime.data(), pointerToVal), static_cast<int>(this->n_total - 1));

                // copy point to save additional information
                // 复制当前点到全局点云中
                globalPoints.points[globalId] = currCloud.points[k];
                ++globalId;
            }
        }
    }

    // 初始化重力方向，用于校正IMU数据或姿态估计
    void initGravityDir()
    {
        // Vector3d measuredGravity = accMeas.rowwise().mean();
        Vector3d measuredGravity = accMeas.col(0);

        Vector3d v1, v2;

        // init gravity
        v1 = gravity;
        v2 = -1.0 * measuredGravity;

        cout << "Measured gravity norm at initialization: " << measuredGravity.norm() << " [m/s2] " << endl;

        // Find the normalized cross product of the two vectors
        Vector3d axis = v1.cross(v2).normalized();

        // Find the angle between the two vectors
        double angle = acos(v1.dot(v2) / (v1.norm() * v2.norm()));

        // Construct the rotation matrix using Rodrigues' rotation formula
        Matrix3d K;
        K << 0.0, -axis.z(), axis.y(),
            axis.z(), 0.0, -axis.x(),
            -axis.y(), axis.x(), 0.0;
        Matrix3d R_to_grav = Matrix3d::Identity() + sin(angle) * K + (1 - cos(angle)) * K * K;

        AngleAxisd angleAxis(R_to_grav.transpose());

        Vector3d normedAxisAngle = angleAxis.angle() * angleAxis.axis();

        controlPoses.relativePoses.Orientations.col(0) = normedAxisAngle;

        // update global params
        controlPoses.relative2global();

        cout << "Estimated gravity direction: " << normedAxisAngle.transpose() << endl;
    }

    // 初始化轨迹数据结构，包括时间间隔、姿态估计、IMU数据和参数估计等
    void initTraj(double t_min, double t_max, int numControlPoses, bool useImu, double dtResIn)
    {

        // init
        dt_res = dtResIn;   // 时间分辨率
        useImuErrorTerms = useImu;  // 是否使用 IMU 误差项

        t0 = t_min;     // 轨迹的起始时间
        horizon = t_max - t_min + dt_res;   // 轨迹的预测范围
        n_total = round(horizon / dt_res) + 1;  // 计算总的时间点数

        // init data structures 初始化变换矩阵结构
        denseTformsLocal2Global.resize(n_total);

        for (int k = 0; k < n_total; ++k)
            denseTformsLocal2Global[k] = Matrix4f::Identity();

        accMeas.conservativeResize(3, n_total);
        angVelMeas.conservativeResize(3, n_total);

        // 初始化全局姿态矩阵
        denseGlobalPoses.resize(n_total);

        // 初始化轨迹时间戳
        trajTime = VectorXd::LinSpaced(n_total, 0.0, horizon);

        // calc total number of parameters
        numParams = numControlPoses;    // 计算总的参数数量

        // set number of parameters 初始化控制姿态
        controlPoses = StampedConsecutivePoses(numParams);

        // init stamps of sparse poses 初始化控制姿态的时间戳
        controlPoses.stamps = VectorXd::LinSpaced(numParams, 0.0, horizon);

        // init param indices
        paramIndices.resize(numParams);
        paramIndices = (controlPoses.stamps / dt_res).array().round().matrix().cast<int>();

        preintImuRots.resize(numParams);
        preintRelPositions.resize(numParams);
        preintRelVelocity.resize(numParams);
        CovPVRot_inv.resize(numParams);

        imuFactorError.resize((numParams - 1));

        gravity << 0.0, 0.0, -9.805;
    }

    // 将一个IMU缓冲区中的测量数据转移到当前的轨迹时间戳上，把测量数据存到 accMeas angVelMeas
    void transferImuMeasurements(ImuBuffer &imuBuffer)
    {

        double timediff;
        // 遍历轨迹中每个时间点
        for (int k = 0; k < n_total; ++k)
        {
            double currGlobalTime = t0 + trajTime(k);
            // 根据当前时间，从 imuBuffer 中找到最近的测量数据，并返回时间差
            timediff = imuBuffer.getClosestMeasurement(currGlobalTime, accMeas.col(k), angVelMeas.col(k));

            if (abs(timediff) > 0.1)
            {
                cout << "Traj timediff to closest imu measurement is: " << timediff << " [s] ! \n"
                     << endl;
            }
        }
    }

    // 更新初始估计，包括重力方向、姿态估计、位置估计等
    void updateInitialGuess(bool &isInitialized, ContinuousTrajectory &oldTraj, bool useImu)
    {
        int lastKnownParamId = 0;
        // 初始化重力方向
        if (!isInitialized)
        {
            // initialize gravity dir
            if (useImu)
                initGravityDir();

            isInitialized = true;

            return;
        }

        // update global parameter set 相对姿态转换为全局姿态
        oldTraj.controlPoses.relative2global();

        // 遍历 controlPoses 中的时间戳，找到最后一个时间戳小于 oldTraj 预测范围的时间戳对应的参数ID
        for (int k = 0; k < controlPoses.stamps.size(); ++k)
        {
            if (t0 + controlPoses.stamps(k) < oldTraj.t0 + oldTraj.horizon)
                lastKnownParamId = k;
        }

        // get orientation for control poses from old trajectory by interpolation
        // 通过插值从 oldTraj 获取控制姿态的方向
        for (int k = 0; k <= lastKnownParamId; ++k)
        {
            getInterpRotation(oldTraj.controlPoses.globalPoses, oldTraj.controlPoses.stamps, controlPoses.stamps(k) + t0 - oldTraj.t0, controlPoses.globalPoses.Orientations.col(k));
        }

        // get position of control poses from old trajectory by cubic interpolation
        Vector3d v0;
        // 通过三次插值获取控制姿态的位置
        for (int k = 0; k < 3; ++k)
        {

            // Map translations to std vec
            VectorXd translations = oldTraj.controlPoses.globalPoses.Translations.row(k);
            Map<VectorXd> translationsMap(translations.data(), translations.size());
            vector<double> vec_translations(translationsMap.data(), translationsMap.data() + translationsMap.size());

            // Map XYZ to std vec
            Map<VectorXd> paramStampsMap(oldTraj.controlPoses.stamps.data(), oldTraj.controlPoses.stamps.size());
            vector<double> vec_paramStamps(paramStampsMap.data(), paramStampsMap.data() + paramStampsMap.size());

            // spline interpolation
            boost::math::barycentric_rational<double> s(vec_paramStamps.data(), vec_translations.data(), vec_paramStamps.size(), 2);

            for (int j = 0; j <= lastKnownParamId; ++j)
                controlPoses.globalPoses.Translations(k, j) = (double)s(controlPoses.stamps(j) + t0 - oldTraj.t0);

            // numeric differentation for velocity
            v0(k) = s.prime(controlPoses.stamps(lastKnownParamId) + t0 - oldTraj.t0);
        }

        // update relative parameter set 全局姿态转换为相对姿态
        controlPoses.global2relative();

        if (useImu)
        {

            Vector3d pos0 = controlPoses.globalPoses.Translations.col(lastKnownParamId);
            Vector3d axang0 = controlPoses.globalPoses.Orientations.col(lastKnownParamId);
            double t0, tend;

            Vector3d axang_end, pos_end, v_end;

            // update params

            for (int k = lastKnownParamId; k < numParams - 1; ++k)
            {
                t0 = controlPoses.stamps(k);

                tend = controlPoses.stamps(k + 1);

                // integrate parameters
                getImuIntegratedParams(t0, axang0, pos0, v0, tend, axang_end, pos_end, v_end);

                // save parameters
                controlPoses.globalPoses.Orientations.col(k + 1) = axang_end;
                controlPoses.globalPoses.Translations.col(k + 1) = pos_end;

                axang0 = axang_end;
                pos0 = pos_end;
                v0 = v_end;
            }

            // update relative parameter set
            controlPoses.global2relative();
        }
        else
        {
            // predict const vel / angular vel
            for (int k = lastKnownParamId; k < numParams - 1; ++k)
            {
                controlPoses.relativePoses.Orientations.col(k + 1) = controlPoses.relativePoses.Orientations.col(lastKnownParamId);
                controlPoses.relativePoses.Translations.col(k + 1) = controlPoses.relativePoses.Translations.col(lastKnownParamId);
            }

            // update relative parameters
            controlPoses.relative2global();
        }
    }

    // 对imu进行积分
    void getImuIntegratedParams(double t0, Vector3d axang0, Vector3d pos0, Vector3d v0,
                                double tend, Ref<Vector3d> axang_end, Ref<Vector3d> pos_end, Ref<Vector3d> v_end)
    {

        // integrate position and orientation within t0 and tend

        // get closest imu measurement index 根据当前时间获得最近时间的索引
        auto pointerToVal = lower_bound(trajTime.data(), trajTime.data() + trajTime.size() - 1, t0);
        auto index = distance(trajTime.data(), pointerToVal);

        double prev = *(pointerToVal - 1);
        double next = *pointerToVal;

        if (abs(t0 - prev) < abs(t0 - next))
            index = index - 1;

        // init variables   初始化
        Matrix3d R_imu2w = axang2rotm(axang0);
        Vector3d vel_w, pos_w;

        pos_w = pos0;
        vel_w = v0;

        double dt_res2 = dt_res * dt_res;
        double currTime = t0;

        // integration loop
        while (abs(currTime + dt_res - tend) < abs(currTime - tend) && index < n_total)
        {
            // update world position
            pos_w = pos_w + vel_w * dt_res + 0.5 * gravity * dt_res2 + 0.5 * R_imu2w * accMeas.col(index) * dt_res2;

            // update world velocity
            vel_w = vel_w + gravity * dt_res + R_imu2w * accMeas.col(index) * dt_res;

            // update rotation
            R_imu2w = R_imu2w * axang2rotm(dt_res * angVelMeas.col(index));

            // update indices
            index += 1;
            currTime += dt_res;
        }

        // save final parameters
        axang_end = rotm2axang(R_imu2w);
        pos_end = pos_w;
        v_end = vel_w;
    }

    void updatePreintFactors(const Ref<Matrix3d> gyr_cov, const Ref<Matrix3d> acc_cov)
    {
        ImuPreintegration imuPreintegration;

        preintImuRots[0] = Matrix3d::Identity();
        preintRelPositions[0].setZero();
        preintRelVelocity[0].setZero();

        int fromId, toId;

        Matrix<double, 9, 9> covMat;

        for (int k = 1; k < controlPoses.numPoses; ++k)
        {
            fromId = paramIndices(k - 1);
            toId = paramIndices(k);

            // reset
            imuPreintegration.reset();

            // preintegration loop
            for (int t = fromId; t < toId; ++t)
            {
                // update preintegration factors
                imuPreintegration.addMeasurement(angVelMeas.col(t), accMeas.col(t), dt_res, gyr_cov, acc_cov);
            }

            // save preint factors
            preintImuRots[k] = imuPreintegration.getDeltaRot();
            preintRelPositions[k] = imuPreintegration.getDeltaPos();
            preintRelVelocity[k] = imuPreintegration.getDeltaVel();
            covMat = imuPreintegration.getCov_rot_v_p();
            CovPVRot_inv[k] = covMat.inverse();
        }

        // preintegrate over complete horizon

        // reset
        imuPreintegration.reset();

        // preintegration loop
        for (int t = 0; t < angVelMeas.cols(); ++t)
        {
            // update preintegration factors
            imuPreintegration.addMeasurement(angVelMeas.col(t), accMeas.col(t), dt_res, gyr_cov, acc_cov);
        }

        preintPosComplHor = imuPreintegration.getDeltaPos();
    }

    void getInterpRotation(const Poses &posesGlobal, const VectorXd &stamps, const double &t, Ref<Vector3d> axangNew)
    {
        // find corresponding index
        auto pointerToVal = lower_bound(stamps.data(), stamps.data() + stamps.size() - 1, t);

        auto rightIndex = distance(stamps.data(), pointerToVal);
        double t_rel;

        if (rightIndex != 0)
            t_rel = (t - stamps(rightIndex - 1)) / (stamps(rightIndex) - stamps(rightIndex - 1));
        else
            t_rel = 1.0f;

        if (rightIndex > 0)
        {
            axangNew = slerp(posesGlobal.Orientations.col(rightIndex - 1), posesGlobal.Orientations.col(rightIndex), t_rel);
        }
        else
        {
            axangNew = posesGlobal.Orientations.col(rightIndex);
        }
    }

    void getSubmapGravityEstimate(Vector3d &gravity_imu)
    {
        // sum up preint positions
        double one_div_t_res = 1.0 / dt_res;
        Vector3d v_start_w = one_div_t_res * (denseGlobalPoses.Translations.col(1) - denseGlobalPoses.Translations.col(0));
        Matrix3d R_imu2w_start = axang2rotm(controlPoses.globalPoses.getFirstOrientation());

        gravity_imu = (R_imu2w_start.transpose() * (controlPoses.globalPoses.getLastTranslation() - controlPoses.globalPoses.getFirstTranslation() - v_start_w * horizon) - preintPosComplHor) / (0.5 * pow(horizon, 2));
    }

    void updateImuError()
    {
        /* preint ACCELERATION POS ERROR */
        controlPoses.global2relative();

        // reset
        imuFactorError.setZero();

        // init
        Vector3d pos_error, rot_error;
        VectorXd combined_error(9);
        combined_error.setZero();

        Vector3d v_start_w, delta_p_model, v_end_world, delta_v_model, vel_error;
        Matrix3d R_imu2w_start;
        Matrix3d R_imu2w_end, R_tmp;
        double delta_t;

        double one_div_t_res = 1.0 / dt_res;

        // cout <<"\n\n"<<endl;

        for (int k = 1; k < paramIndices.size(); ++k)
        {

            // calculate position error
            R_imu2w_start = axang2rotm(controlPoses.globalPoses.Orientations.col(k - 1));

            delta_t = controlPoses.stamps(k) - controlPoses.stamps(k - 1);

            // calc start velocity
            v_start_w = one_div_t_res * (denseGlobalPoses.Translations.col(paramIndices(k - 1) + 1) - denseGlobalPoses.Translations.col(paramIndices(k - 1)));

            // calc end velocity
            v_end_world = one_div_t_res * (denseGlobalPoses.Translations.col(paramIndices(k)) - denseGlobalPoses.Translations.col(paramIndices(k) - 1));

            delta_p_model = R_imu2w_start.transpose() * (controlPoses.globalPoses.Translations.col(k) - controlPoses.globalPoses.Translations.col(k - 1) - v_start_w * delta_t - 0.5 * pow(delta_t, 2) * gravity);

            // + R_imu2w_start*
            pos_error = delta_p_model - preintRelPositions[k];

            // calculate rotation error
            R_imu2w_end = axang2rotm(controlPoses.relativePoses.Orientations.col(k));
            R_tmp = preintImuRots[k].transpose() * R_imu2w_end;

            rot_error = rotm2axang(R_tmp);

            // calculate velocity error
            delta_v_model = R_imu2w_start.transpose() * (v_end_world - v_start_w - gravity * delta_t);

            vel_error = delta_v_model - preintRelVelocity[k];

            // save error
            combined_error << rot_error, vel_error, pos_error;

            // save
            imuFactorError(k - 1) = combined_error.transpose() * CovPVRot_inv[k] * combined_error;

            imuFactorError(k - 1) *= balancingImu;
        }
    }

    const int &getIdOfPoint(int &globalPointIndex)
    {
        return globalPoints.points[globalPointIndex].id;
    }
};

#endif
