/* Copyright (C) 2024 David Skuddis - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: davidskuddis@web.de, or visit: https://opensource.org/license/mit/
 */

#ifndef DMSASLAM_H
#define DMSASLAM_H

#include <eigen3/Eigen/Core>
#include <memory>
#include <chrono>
#include "ImuBuffer.h"
#include "PointCloudBuffer.h"
#include "Config.h"
#include "ContinuousTrajectory.h"
#include "MapManagement.h"
#include "DmsaOptimizer.h"
#include "helpers.h"
#include "OutputManagement.h"

using namespace pcl;
using namespace std;

class DmsaSlam
{
public:
    ImuBuffer imuBuffer;
    std::shared_ptr<PointCloudBuffer> pcBuffer;

    bool oneCloudBufferInit = false;        // 点云缓冲区是否初始化
    PointCloudPlus::Ptr bufferedCloud;      // 缓冲的每一帧点云

    // state
    bool submapIsInitialized = false;   // 子地图是否初始化
    bool recievedImuData = false;       // 是否接收到过 IMU 数据
    bool timeInitialized = false;
    double t0 = -1.0;                   // 点云中最小的时间戳
    chrono::time_point<std::chrono::system_clock> t0_system;    // 找到最小时间戳时的系统时间

    std::shared_ptr<ContinuousTrajectory> currTraj;     // 当前轨迹
    std::shared_ptr<ContinuousTrajectory> oldTraj;

    MapManagement KeyframeMap;      // 地图管理

    Config config;
    DmsaOptimSettings optimSettingsSlidingWindow;   // 滑动窗口优化的设置
    DmsaOptimSettings optimSettingsMap;             // 地图优化的设置

    DmsaOptimizer<PointStampId> slidingWindowOptimizer;
    DmsaOptimizer<PointNormal> keyframeMapOptimizer;

    PointCloud<PointStampId>::Ptr staticPoints;

    int maxOverlapKeyId = 0;    // 最大重叠关键帧 ID
    float overlapRatio = 0.0f;  // 重叠比例

    OutputManagement Output;

    DmsaSlam(Config &inputConf) : config(inputConf), currTraj(new ContinuousTrajectory()), oldTraj(new ContinuousTrajectory()), pcBuffer(new PointCloudBuffer()), staticPoints(new PointCloud<PointStampId>())
    {
        pcBuffer->init(config.n_clouds);

        KeyframeMap = MapManagement(config.last_n_keyframes_for_optim);

        staticPoints->reserve(config.expected_max_num_static_pts);

        initConfig();
    }

    DmsaSlam() : currTraj(new ContinuousTrajectory()), oldTraj(new ContinuousTrajectory()), pcBuffer(new PointCloudBuffer()), staticPoints(new PointCloud<PointStampId>())
    {
        pcBuffer->init(config.n_clouds);

        KeyframeMap = MapManagement(config.last_n_keyframes_for_optim);

        staticPoints->reserve(config.expected_max_num_static_pts);

        initConfig();
    }

    void initConfig()
    {
        optimSettingsSlidingWindow.num_iter = config.num_iter_sliding_window_optim;         // 滑动窗口优化迭代次数
        optimSettingsSlidingWindow.min_num_points_per_set = config.min_num_points_gauss;    // 最小点数阈值
        optimSettingsSlidingWindow.decay_rate = config.decay_rate_sw;                       // 衰减率

        optimSettingsMap.min_num_points_per_set = config.min_num_points_gauss;  // 最小点数阈值
        optimSettingsMap.step_length_optim = config.alpha_keyframe_optim;       // 步长
        optimSettingsMap.epsilon = config.epsilon_keyframe_opt;                 // 优化的精度阈值

        optimSettingsMap.gauss_split = true;                            // 是否高斯分裂
        optimSettingsMap.num_iter = config.num_iter_keyframe_optim;     // 迭代次数
        optimSettingsMap.max_step = 0.01;                               // 最大步长
        // optimSettingsMap.step_length_optim = 0.05;
        optimSettingsMap.decay_rate = config.decay_rate_key;            // 衰减率

        optimSettingsMap.select_best_set = config.select_best_set_key;
        optimSettingsMap.min_num_gaussians = config.min_num_points_gauss_key;
        optimSettingsMap.grid_size_1_factor = 1.5f;
        optimSettingsMap.grid_size_1_factor = 2.0f;
    }

    // 把 IMU 数据加入缓存 imuBuffer
    void processImuMeasurements(Eigen::Vector3d &AccMeas, Eigen::Vector3d &AngVelMeas, double &stamp)
    {
        // init t0
        // 检查系统时间是否初始化，如果没有初始化，放弃掉这一帧 IMU
        if (timeInitialized == false)
        {
            std::cerr << "Discard imu data because the system is only initialized with the first point cloud\n";
            return;
        }

        if (!recievedImuData)
            recievedImuData = true;

        imuBuffer.addMeasurement(AccMeas, AngVelMeas, stamp + config.timeshift_to_imu);
    }

    void processPointCloud(PointCloudPlus::Ptr inputPc)
    {
        // 更新时间管理信息
        updateTimeManagement(inputPc);

        // 初始化缓冲区，第一次调用 点云赋值给 bufferedCloud
        if (oneCloudBufferInit == false)
        {
            bufferedCloud = PointCloudPlus::Ptr(inputPc);
            oneCloudBufferInit = true;
            std::cerr << "One cloud buffer is initialized "
                      << "\n";
            return;
        }

        // 交换缓冲区
        PointCloudPlus::Ptr pcToProcess(bufferedCloud);
        bufferedCloud = PointCloudPlus::Ptr(inputPc);

        // PREPROCESSING 预处理
        PointCloudPlus::Ptr filteredPc(new PointCloudPlus);

        preProcess(pcToProcess, filteredPc);

        // UPDATE RINGBUFFER    更新环形缓冲区
        pcBuffer->addElem(*filteredPc);

        // stop here if the buffer is not yet full
        // 检查点云环形缓冲区满没满，没满直接 return
        if (pcBuffer->isFull() == false)
        {
            std::cerr << "Point cloud ring buffer is not full yet " << pcBuffer->getNumUpdates() << " / " << pcBuffer->getMaxNumElems() << "\n";
            return;
        }

        // SUBMAP OPTIMIZATION 准备轨迹优化，更新当前轨迹
        prepareTrajectoryForOptimization();

        // INIT MAP
        if (KeyframeMap.isInitialized == false)
        {
            initializeMap(*currTraj);
            return;
        }

        // add static points 添加静态点最小关键帧 ID
        int minKeyframeId;

        // find relevant static points for sliding window optim, calc overlap, find minimum related keyframe
        // 找到与滑动窗口优化相关的静态点，计算重叠部分，并找到相关的最小关键帧ID
        addStaticPoints(*currTraj, maxOverlapKeyId, overlapRatio, minKeyframeId);

        // optimize submap 滑动窗口优化子地图
        slidingWindowOptimizer.optimizeSet(*currTraj, optimSettingsSlidingWindow);
        // 移除静态点
        currTraj->removeStaticPoints();

        Ref<Vector3d> lastKeyframePos = KeyframeMap.keyframePoses.globalPoses.Translations.col(KeyframeMap.keyframeDataBuffer.getNumElements() - 1);
        Ref<Vector3d> currPos = currTraj->controlPoses.globalPoses.Translations.col(0);

        // 根据重叠比例和距离判断是否添加新关键帧
        if (overlapRatio < config.min_overlap_new_keyframe || (currPos - lastKeyframePos).norm() > config.dist_new_keyframe)
        {
            if (KeyframeMap.keyframeDataBuffer.isFull() == true)
                --minKeyframeId;

            std::cerr << "Add keyframe no. " << KeyframeMap.keyframeDataBuffer.getNumUpdates() << " Overlap: " << overlapRatio << "\n";
            addNewKeyframeToMap(*currTraj);

            std::cerr << "Keyframe optimization from " << minKeyframeId << " to end\n";
            if (config.optimize_sliding_window_keyframes)
                keyframeOptimization(minKeyframeId, KeyframeMap);

            KeyframeMap.updateGlobalPoints();
        }
        else
        {
            ConsecutivePoses tmp(2);

            tmp.globalPoses.Translations.col(0) = KeyframeMap.keyframePoses.globalPoses.Translations.col(maxOverlapKeyId);
            tmp.globalPoses.Orientations.col(0) = KeyframeMap.keyframePoses.globalPoses.Orientations.col(maxOverlapKeyId);

            tmp.globalPoses.Translations.col(1) = currTraj->controlPoses.globalPoses.Translations.col(0);
            tmp.globalPoses.Orientations.col(1) = currTraj->controlPoses.globalPoses.Orientations.col(0);

            tmp.global2relative();

            Output.addNonKeyframePose(tmp.relativePoses.Translations.col(1), tmp.relativePoses.Orientations.col(1), currTraj->t0, maxOverlapKeyId);
        }

        // END: reset imu flag to handle temporary lack of imu data
        recievedImuData = false;
    }

    // 保存 pose 到文件
    void savePoses(string result_dir)
    {
        Output.saveDensePoses(KeyframeMap.keyframePoses, result_dir);
    }

private:
    void keyframeOptimization(int &fromId, MapManagement &allKeyframes)
    {
        if (fromId < 0 || allKeyframes.keyframeDataBuffer.getMaxNumElems() == 1 || allKeyframes.keyframeDataBuffer.getMaxNumElems() < 3)
            return;

        // 根据给定索引 fromId 到 allKeyframes.keyframeDataBuffer.getNumElements() - 1，创建子地图
        std::shared_ptr<MapManagement> currSubmap = allKeyframes.getSubmap(fromId, allKeyframes.keyframeDataBuffer.getNumElements() - 1);

        // copy settings 对子地图进行一些设置
        currSubmap->useGravityErrorTerms = config.use_gravity_term_in_keyframe_opt && config.use_imu;
        currSubmap->balancingFactorGrav = config.balancing_factor_gravity;

        currSubmap->useOdometryErrorTerms = config.use_odometry_term_in_keyframe_opt;
        currSubmap->balancingFactorOdom = config.balancing_factor_odometry;

        // keyframe optimization 对子地图关键帧进行优化
        std::cout << "Grid size keyframe optimization: " << currSubmap->minGridSize << std::endl;
        keyframeMapOptimizer.optimizeSet(*currSubmap, optimSettingsMap);

        // update poses 更新位姿，把子地图中的位姿放入 allKeyframes
        allKeyframes.updatePosesFromSubmap(fromId, allKeyframes.keyframeDataBuffer.getNumElements() - 1, *currSubmap);

        // update curr trajectory   更新当前轨迹
        currTraj->controlPoses.relativePoses.Translations.col(0) = allKeyframes.keyframePoses.globalPoses.Translations.col(allKeyframes.keyframeDataBuffer.getNumElements() - 1);
        currTraj->controlPoses.relativePoses.Orientations.col(0) = allKeyframes.keyframePoses.globalPoses.Orientations.col(allKeyframes.keyframeDataBuffer.getNumElements() - 1);

        // 优化后，相对位姿转换为全局位姿
        currTraj->controlPoses.relative2global();
    }

    void updateTimeManagement(PointCloudPlus::Ptr inputPc)
    {
        // 如果时间还未初始化
        if (timeInitialized == false)
        {
            // 遍历点云，找到最小的时间戳
            double minStamp = std::numeric_limits<double>::max();
            for (auto point : inputPc->points)
                minStamp = std::min(minStamp, point.stamp);

            t0 = minStamp;
            t0_system = std::chrono::system_clock::now();

            timeInitialized = true;
            std::cerr << "Time is initialized. t0 = " << t0 << "\n";
        }

        chrono::time_point<std::chrono::system_clock> currSystemTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds_system = currSystemTime - t0_system;

        // 计算输入点云中第一个点的时间戳与初始时间戳t0的差值elapsedSecDataTime，表示数据时间的流逝
        double elapsedSecDataTime = inputPc->points[0].stamp - t0;

        // 如果点云缓冲区的更新次数是 10 的倍数，输出处理的实时时间elapsedSecDataTime以及数据时间与系统时间的比率。这个比率可以用来评估数据处理的速度与实际时间的关系
        if (pcBuffer->getNumUpdates() % 10 == 0)
            std::cerr << "Processed real-time: " << elapsedSecDataTime << " [s] / real-time ratio: " << elapsedSecDataTime / elapsed_seconds_system.count() << "\n";
    }

    // 将关键帧中符合条件的点加入静态点，并找出重叠度最高的关键帧ID
    // 计算关键帧与当前轨迹位置的距离。
    // 如果距离在阈值内，则处理关键帧的点云数据。
    // 对于每个点，使用Kd树找到最近的邻居，并检查它是否在允许的距离内且对当前轨迹可见。
    // 如果是，则将点添加到静态点集合中，并更新重叠度。
    // 最后，函数对静态点进行下采样，计算重叠度，并将活动点云作为静态点添加到轨迹中。
    void addStaticPoints(ContinuousTrajectory &trajIn, int &keyframeId, float &overlapToStatic, int &minRelatedKeyId)
    {
        // 当前轨迹的位置
        Ref<Vector3d> currPos = trajIn.controlPoses.globalPoses.Translations.col(0);
        Vector3f currPosf = currPos.cast<float>();

        keyframeId = 0;
        overlapToStatic = 0.0f; // 当前轨迹与静态点的重叠率
        int currOverlap;
        int maxOverlapKey = 0;
        minRelatedKeyId = -1;

        PointCloud<PointNormal>::Ptr tmpPtr(new PointCloud<PointNormal>());

        PointStampId tmpPt;
        tmpPt.id = -1;
        tmpPt.stamp = -1000.0;

        // reset static points  重置静态点
        staticPoints->resize(0);

        KdTreeFLANN<PointStampId> kdtree;

        // Set the input point cloud to the kdtree
        // 使用当前轨迹的全局点云数据初始化 kdtree
        kdtree.setInputCloud(trajIn.globalPoints.makeShared());

        vector<int> pointIdxNKNSearch;          // 最近点的索引
        vector<float> pointNKNSquaredDistance;  // 最近点的距离

        float sqrdMaxDist = std::pow(1.0f * trajIn.minGridSize, 2);

        for (int k = 0; k < KeyframeMap.keyframeDataBuffer.getNumElements(); ++k)
        {
            if (k < KeyframeMap.keyframeDataBuffer.getNumElements() - config.oldest_k_keyframes_as_static_points)
            {
                continue;
            }
            // 当前轨迹与关键帧轨迹的距离
            double distToKeyframe = (currPos - KeyframeMap.keyframePoses.globalPoses.Translations.col(k)).norm();

            if (distToKeyframe < config.dist_static_points_keyframe)
            {
                KeyframeMap.getGlobalKeyframeCloud(k, tmpPtr);

                // reset overlap 重置重叠率
                currOverlap = 0;

                // for (auto point : tmpPtr->points)
                // 遍历关键帧点云，搜索最近邻更新静态点
                for (int j = 0; j < tmpPtr->points.size(); ++j)
                {
                    auto &point = tmpPtr->points[j];

                    tmpPt.getVector3fMap() = point.getVector3fMap();

                    if (kdtree.nearestKSearch(tmpPt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        if (pointNKNSquaredDistance[0] <= sqrdMaxDist)
                        {
                            // check visibility
                            if (isVisible(currPosf, point))
                            {
                                tmpPt.id = KeyframeMap.keyframeDataBuffer.at(k).ringIds(j);

                                staticPoints->push_back(tmpPt);

                                // update overlap
                                ++currOverlap;

                                if (minRelatedKeyId < 0)
                                    minRelatedKeyId = k;
                            }
                        }
                    }

                    // save keyframe with maximum overlap
                    if (currOverlap > maxOverlapKey)
                    {
                        maxOverlapKey = currOverlap;
                        keyframeId = k;
                    }
                }
            }
        }

        // downsampling 静态点下采样到 KeyframeMap.activePoints
        if (staticPoints->size() > 0)
            randomGridDownsampling(staticPoints, KeyframeMap.activePoints, trajIn.minGridSize / 2.0f);

        if (pcBuffer->getNumUpdates() % 10 == 0)
            std::cerr << "num pts active: " << KeyframeMap.activePoints->size() << " Mapsize: " << KeyframeMap.keyframeDataBuffer.getNumUpdates() << " / max: " << KeyframeMap.keyframeDataBuffer.getMaxNumElems() << "\n";

        // 计算静态点与轨迹全局点云之间的重叠度
        overlapToStatic = getOverlap(*KeyframeMap.activePoints, trajIn.globalPoints, trajIn.minGridSize);

        // 下采样后的静态点加入当前轨迹
        if (KeyframeMap.activePoints->size() > 0)
            trajIn.addStaticPoints(*KeyframeMap.activePoints);
    }

    bool isVisible(Vector3f &pos, PointNormal &point)
    {
        float res;
        float d;

        d = point.getVector3fMap().transpose() * point.getNormalVector3fMap();

        // check visibility
        res = pos.transpose() * point.getNormalVector3fMap() - d;

        // check if residuum is greater or equal to zero
        if (res >= -0.00001)
            return true;
        else
            return false;
    }

    template <typename PointType1, typename PointType2>
    float getOverlap(PointCloud<PointType1> &pc1, PointCloud<PointType2> &pc2, float maxDistOverlap)
    {
        if (pc1.size() == 0 || pc2.size() == 0)
            return 0.0f;

        // octree::OctreePointCloudSearch<PointType1> octree(maxDistOverlap);
        KdTreeFLANN<PointType1> kdtree;

        float sqrdMaxDist = maxDistOverlap * maxDistOverlap;

        // Set the input point cloud to the octree
        kdtree.setInputCloud(pc1.makeShared());

        // Construct the octree
        // octree.addPointsFromInputCloud();

        PointType1 searchPoint;

        vector<int> pointIdxNKNSearch;

        vector<float> pointNKNSquaredDistance;

        int nCorresp = 0;

        for (auto point : pc2.points)
        {
            searchPoint.getVector3fMap() = point.getVector3fMap();

            if (kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                if (pointNKNSquaredDistance[0] <= sqrdMaxDist)
                    ++nCorresp;
            }
        }

        return static_cast<float>(nCorresp) / static_cast<float>(pc2.size());
    }

    // 准备轨迹优化
    // 把旧轨迹交换到当前轨迹
    // 将 imu 数据转换到当前轨迹，并更新轨迹中的预积分因子
    // 把点云缓冲区注册到当前轨迹
    void prepareTrajectoryForOptimization()
    {
        // save old Traj 将旧轨迹交换到当前轨迹
        currTraj.swap(oldTraj);

        double minStamp, maxStamp;
        // 获取最大最小时间戳
        pcBuffer->getMinMaxPointStamps(minStamp, maxStamp);

        // init trajectory
        bool useImuNow = config.use_imu && recievedImuData;
        // 初始化轨迹
        currTraj->initTraj(minStamp, maxStamp, config.num_control_poses, useImuNow, config.dt_res);

        // deactivate imu usage for the whole sequence if there are no measurements before initialization
        if (submapIsInitialized == false && config.use_imu == true && recievedImuData == false)
        {
            std::cerr << "Usage of imu data is activated in the config, but it will not be used because no imu data arrived before initialization . . . \n";
            config.use_imu = false;
        }

        // add corresponding imu measurements to trajectory
        // 把 imu 数据转移到当前轨迹中
        if (useImuNow)
            currTraj->transferImuMeasurements(imuBuffer);

        // update preintegrated rotations
        // 更新当前轨迹中的预积分旋转因子
        if (useImuNow)
            currTraj->updatePreintFactors(config.cov_gyr, config.cov_acc);

        // update initial guess 更新初始估计
        currTraj->updateInitialGuess(submapIsInitialized, *oldTraj, useImuNow);

        // register pc buffer   点云缓冲区注册到当前轨迹
        currTraj->registerPcBuffer(pcBuffer);

        // 更新当前轨迹的全局点
        currTraj->updateGlobalPoints();

        currTraj->validImuData = useImuNow;

        if (config.use_imu && recievedImuData)
        {
            // 设置滑动窗口优化的步长和最大步长参数
            optimSettingsSlidingWindow.step_length_optim = config.alpha_sliding_window_imu;
            optimSettingsSlidingWindow.max_step = config.max_step_sliding_window_imu;

            currTraj->balancingImu = config.imu_factor_weight_submap;
        }
        else
        {
            optimSettingsSlidingWindow.step_length_optim = config.alpha_sliding_window_no_imu;
            optimSettingsSlidingWindow.max_step = config.max_step_sliding_window_no_imu;
            // optimSettingsSlidingWindow.decay_rate = 1.0;
        }
    }

    // 初始化地图，将轨迹中点云缓冲区的第一个点云加入到关键帧地图
    void initializeMap(ContinuousTrajectory &trajIn)
    {
        PointCloud<PointNormal>::Ptr keyframeCloud_imu(new PointCloud<PointNormal>());

        keyframeCloud_imu->resize(trajIn.regPcBuffer->at(0).size());

        KeyframeData data;
        data.ringIds.resize(trajIn.regPcBuffer->at(0).size());
        // 把点云缓冲区中第一个点云信息复制到 keyframeCloud_imu 和 data
        for (int k = 0; k < trajIn.regPcBuffer->at(0).size(); ++k)
        {
            keyframeCloud_imu->points[k].x = trajIn.regPcBuffer->at(0).points[k].x;
            keyframeCloud_imu->points[k].y = trajIn.regPcBuffer->at(0).points[k].y;
            keyframeCloud_imu->points[k].z = trajIn.regPcBuffer->at(0).points[k].z;

            data.ringIds(k) = trajIn.regPcBuffer->at(0).points[k].id;
        }

        updateNormals(keyframeCloud_imu);

        data.pointCloudLocal = keyframeCloud_imu;
        data.gridSize = trajIn.minGridSize;

        if (trajIn.validImuData)
            // 获取子地图的重力估计值
            trajIn.getSubmapGravityEstimate(data.measuredGravity);

        // 将关键帧添加到地图中
        KeyframeMap.addKeyframe(trajIn.controlPoses.globalPoses.Translations.col(0), trajIn.controlPoses.globalPoses.Orientations.col(0), trajIn.t0, data);

        Output.informAboutNewKeyframe();
    }

    void addNewKeyframeToMap(ContinuousTrajectory &trajIn)
    {
        Output.informAboutNewKeyframe();

        PointCloud<PointStampId>::Ptr keyframeCloudFiltered(new PointCloud<PointStampId>());

        randomGridDownsampling(trajIn.globalPoints.makeShared(), keyframeCloudFiltered, trajIn.minGridSize);

        PointCloud<PointNormal>::Ptr keyframeCloud_imu(new PointCloud<PointNormal>());

        keyframeCloud_imu->resize(keyframeCloudFiltered->size());

        Vector3f currWorldPose = trajIn.controlPoses.globalPoses.Translations.col(0).cast<float>();
        Matrix3f currRotInv = axang2rotm(trajIn.controlPoses.globalPoses.Orientations.col(0)).transpose().cast<float>();

        KeyframeData data;

        data.ringIds.resize(keyframeCloudFiltered->size());

        for (int k = 0; k < keyframeCloudFiltered->size(); ++k)
        {
            keyframeCloud_imu->points[k].getVector3fMap() = currRotInv * (keyframeCloudFiltered->points[k].getVector3fMap() - currWorldPose);

            data.ringIds(k) = keyframeCloudFiltered->points[k].id;
        }

        updateNormals(keyframeCloud_imu);

        data.pointCloudLocal = keyframeCloud_imu;

        // data.pointCloudGlobal=PointCloud<PointNormal>::Ptr(new PointCloud<PointNormal>() );

        data.gridSize = trajIn.minGridSize;

        if (trajIn.validImuData)
            trajIn.getSubmapGravityEstimate(data.measuredGravity);

        // check gravity plausability
        if (std::abs(data.measuredGravity.norm() - KeyframeMap.gravity.norm()) < config.gravity_outlier_thresh)
            data.gravityPlausible = true;
        else
            data.gravityPlausible = false;

        if (trajIn.validImuData == true && data.gravityPlausible == false)
            std::cout << "Discarded faulty gravity measurement . . . " << std::endl;

        std::cout << "Gravity estimation keyframe: " << (axang2rotm(trajIn.controlPoses.globalPoses.Orientations.col(0)) * data.measuredGravity).transpose() << std::endl;

        // save oldest keyframe to output manager
        if (KeyframeMap.keyframeDataBuffer.isFull())
        {
            Output.addStaticKeyframePose(KeyframeMap.keyframePoses.globalPoses.Translations.col(0), KeyframeMap.keyframePoses.globalPoses.Orientations.col(0), KeyframeMap.keyframePoses.stamps(0));
        }

        KeyframeMap.addKeyframe(trajIn.controlPoses.globalPoses.Translations.col(0), trajIn.controlPoses.globalPoses.Orientations.col(0), trajIn.t0, data);
    }

    void updateNormals(PointCloud<PointNormal>::Ptr &cloud, Vector3d origin = Vector3d::Zero())
    {
        NormalEstimationOMP<PointNormal, PointNormal> ne;

        search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>);
        tree->setInputCloud(cloud);
        ne.setInputCloud(cloud);
        ne.setSearchMethod(tree);
        ne.setKSearch(6);
        ne.setViewPoint(origin(0), origin(1), origin(2));
        ne.compute(*cloud);
    }

    void preProcess(PointCloudPlus::Ptr rawPc, PointCloudPlus::Ptr filteredPc)
    {
        // adaptive random grid filter
        filteredPc->gridSize = 0.4f;
        randomGridDownsampling(rawPc, filteredPc, 0.4f);

        if (filteredPc->size() < config.max_num_points_per_scan)
        {
            filteredPc->gridSize = 0.3f;
            randomGridDownsampling(rawPc, filteredPc, 0.3f);
        }

        if (filteredPc->size() < config.max_num_points_per_scan)
        {
            filteredPc->gridSize = 0.2f;
            randomGridDownsampling(rawPc, filteredPc, 0.2f);
        }

        if (filteredPc->size() < config.max_num_points_per_scan)
        {
            filteredPc->gridSize = 0.15f;
            randomGridDownsampling(rawPc, filteredPc, 0.15f);
        }

        // sort points acc. to range
        std::vector<float> ranges, rangesSorted;

        ranges.resize(filteredPc->size());
        rangesSorted.resize(filteredPc->size());

        for (int k = 0; k < filteredPc->size(); ++k)
        {
            ranges[k] = Eigen::Vector3f(filteredPc->at(k).x, filteredPc->at(k).y, filteredPc->at(k).z).norm();
            rangesSorted[k] = ranges[k];
        }

        sort(rangesSorted.begin(), rangesSorted.end());

        // find threshold range
        float thresRange = std::max(rangesSorted[std::min(config.max_num_points_per_scan, (int)(rangesSorted.size() - 1))], config.minDistDS);

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ExtractIndices<PointStampId> extract;
        for (int i = 0; i < filteredPc->size(); i++)
        {

            if (ranges[i] < thresRange && ranges[i] > config.min_dist)
            {
                inliers->indices.push_back(i);
            }
        }
        extract.setInputCloud(filteredPc);
        extract.setIndices(inliers);
        extract.filter(*filteredPc);

        // transform to imu frame
        pcl::transformPointCloud(*filteredPc, *filteredPc, config.lidarToImuTform);

        // set padding variable to 1
        for (auto &point : filteredPc->points)
            point.data[3] = 1.0f;

        if (pcBuffer->getNumUpdates() % 10 == 0)
            std::cerr << "Grid size preprocessing: " << filteredPc->gridSize << " / num points: " << filteredPc->size() << "\n";
    }
};

#endif
