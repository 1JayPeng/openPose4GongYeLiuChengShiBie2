# -*- coding: utf-8 -*-
import os
import pickle
# import signal
import sys
import time
import traceback
from sys import platform

import cv2
import numpy as np
import yaml

# # Disable HDF5 version check
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

with open('config.yml', 'r', encoding='utf-8') as y:
    dataInQt5Window2 = yaml.safe_load(y)
# 定义保存状态的文件路径
state_file = dataInQt5Window2.get('state_file')


# def sigHandler(signum, frame, state):
#     """
#     信号处理函数，用于捕获程序中断信号并退出程序
#
#     :param signum: 信号编号
#     :param frame: 信号帧
#     :param state: 程序状态
#     """
#
#     usedtimeSec = time.perf_counter() - state['start']
#     state['start'] += usedtimeSec
#     with open(state_file, 'wb') as f:
#         pickle.dump(state, f)
#     traceback.print_exc()
#     raise SystemExit(signum)


def detect(imgPath, model_folder, start1, GPU=1, isShow=False):
    # Set CUDA_VISIBLE_DEVICES to use GPU 1
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    """
    用OpenPose对给定的图像进行人体姿态估计

    Args:
        imgPath (str): 图像路径
        model_folder (str): 模型文件夹路径
        isShow (bool, optional): 是否显示图像. Defaults to False.

    Returns:
        dict: 包含人体姿态信息的字典
    """

    # 获取当前该文件所在文件夹的绝对路径
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('openposeCNN/build/python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the
            # OpenPose/python module from there. This will install OpenPose and the python library at your desired
            # installation path. Ensure that this is in your python path in order to use it. sys.path.append(
            # '/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
            'script in the right folder?')
        raise e
    # 模型导入
    params = dict()
    params['model_folder'] = model_folder  # 模型地址
    params["logging_level"] = 4  # 日志等级
    params["disable_multi_thread"] = False  # 禁用多线程
    # params[
    # "net_resolution"] = '1280x720'  # 分辨率,需要是16的倍数，降低这个参数可以以准确率为代价显著提升处理速度。
    params["net_resolution"] = '320x240'
    params["number_people_max"] = 1

    params["body"] = 1  # 0禁用身体检测，1启用

    params["disable_blending"] = False  # 如果为True，只显示骨骼关键点，背景为黑

    params["model_pose"] = "BODY_25"  # 参数设置"BODY_25“表示使用25点的检测模式，CUDA
    # 版本中最快最准的模式。此外设置"COCO"使用18
    # 点的检测模式，设置"MPI"使用15点的检测模式，最不精确，但在CPU上最快。设置"MPI_4_layers"使用15点的检测模式，甚至比上一种更快，但不够准确。

    params["keypoint_scale"] = 0  # 最终姿态数据数组(x,y)坐标的缩放，即(x,y)的缩放。"将以' write_json '和'

    # write_keypoint '标记保存的坐标。"选择“0”将其缩放到原始源分辨率;' 1 '将其缩放到净输出" "大小(用'
    # net_resolution '设置);' 2 '将其缩放到最终的输出大小(设置为" " '分辨率');' 3 '将其缩放到[0,1]的范围内，其中(
    # 0,0)将是图像左上角的“”角，(1,1)是右下角的“”角;4表示范围[-1,1]，其中" "(-1，-1)是图像的左上角，(1,
    # 1)是右下角。与“scale_number”和“scale_gap”相关的非。
    # signal.signal(signal.SIGABRT, sigHandler, start1)
    try:
        # 启动pyopenpose:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        # 传入一张图片
        datum = op.Datum()

        imageToProcess = cv2.imread(imgPath, cv2.IMREAD_ANYCOLOR)

        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        pose = datum.poseKeypoints
    except:
        usedtimeSec = time.perf_counter() - start1['start']
        start1['start'] += usedtimeSec
        with open(state_file, 'wb') as f:
            pickle.dump(start1, f)
        traceback.print_exc()
        sys.exit(1)  # 退出程序

    # print(hand)
    if isShow:
        try:
            # print(type(pose))
            # print(datum.cvOutputData)
            # print("Body keypoints: \n" + str(pose))
            print(pose.shape)
            for jg in pose:
                print('pose[4]:' + str(jg[4]))
                print("pose[3]:" + str(jg[3]))
                print('pose[6]:' + str(jg[6]))
                print('pose[7]:' + str(jg[7]))
                print()
            # print(pose[0][4][0])
        except AttributeError:
            print("没有检测出人体")
        with open('pose.csv', 'w') as file:
            for b1 in pose:
                np.savetxt(file, b1, delimiter=",")
        # b,g,r = cv2.split(datum.cvOutputData)#分别提取B、G、R通道
        # img= cv2.merge([r,g,b])	#重新组合为R、G、B
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        cv2.namedWindow('opStudy', 0)

        cv2.resizeWindow("opStudy", 600, 600)
        cv2.moveWindow("opStudy", 100, 100)
        cv2.imshow("opStudy", datum.cvOutputData)
        cv2.waitKey(0)
    return pose, datum.cvOutputData
