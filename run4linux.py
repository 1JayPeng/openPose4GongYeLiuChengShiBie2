# -*- coding: utf-8 -*-
import functools
import os
import pickle
import re
import shutil
import signal
import time
import traceback
from multiprocessing import Process

import cv2
import yaml

import Rule
import match
import save
import twiceDetect4linux as twiceDetect

videos = 'videos/'


# num0 = 0
# num1 = 0
# num2 = 0
#
# list0 = []
# list1 = []
# list2 = []
# right = ['1234/']
# lack = ['12/', '13/', '14/', '123/', '124/', '134/']
# outOForder = ['1423/', '1432/', '1342/', '1324/']
# framesl = 0


# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')


def run(videoPath, model_folder, gpu):
    """
    运行程序，检测视频中的操作流程是否正确

    Args:
        videoPath: 视频路径
        model_folder: 模板图片路径
        l: 进程锁
        gpu: 是否使用GPU加速

    Returns:
        无返回值
    """

    start = time.perf_counter()
    flag = 0  # 0 表示正确，1表示 缺项 ，2表示乱序
    # 读取yaml配置文件
    with open('config.yml', 'r', encoding='utf-8') as y:
        dataInQt5Window2 = yaml.safe_load(y)

    templePath = dataInQt5Window2.get('templePath')
    # videoPath = dataInQt5Window2.get('videoPath')
    resultPath = dataInQt5Window2.get('resultPath')

    # 参数 YU_ZHI 当匹配率小于YU_ZHI时视为有意义 数值越高精准度越低 最高为0.99
    # NUM 只有连续NUM帧有意义时才记录帧数  数值越高精准度越高 最低为1
    # LUAN_XU 容错，当后续步骤超过该值时判定乱序  数值越高进准度越低 最低为0
    # 0.95 3 3 时检测步骤1不会出现乱序 当luan_xu低于3时会出现乱序错误
    # 参数 WUCHA 表示当差值低于这个值之后判定在一条线上
    YU_ZHI = dataInQt5Window2.get('YU_ZHI')
    # NUM = dataInQt5Window2.get('NUM')
    LUAN_XU = dataInQt5Window2.get('LUAN_XU')
    WUCHA = dataInQt5Window2.get('WUCHA')
    # ZLX = dataInQt5Window2.get('ZLX')
    # 检测模板图片 模板图片需按正确流程顺序命名 第一个步骤命名为1或temple1
    temples_list = os.listdir(templePath)
    temples_list.sort()

    # 生成保存路径
    i = 1
    savePath = resultPath + "/" + str(i)

    try:
        while os.path.exists(savePath):
            i = i + 1
            savePath = resultPath + "/" + str(i)
        else:

            os.makedirs(savePath)
    finally:
        pass

        # 生成match保存路径
        matchPath = savePath + "/match"
        # i = 1
        # matchPath = savePath + "/" + str(i)
        # while os.path.exists(matchPath):
        #     i = i + 1
        #     matchPath = savePath + "/" + str(i)
        # else:
        #     os.makedirs(matchPath)

    images_dirs = savePath + "/iamges"

    try:
        os.makedirs(images_dirs)
    finally:
        pass

    re_string = "[0-9]*.jpg"
    r = re.compile(re_string)

    # 记录上一次有意义帧的序号
    frame_temp_list = [0, 0, 0, 0]

    # 暂存步骤执行的帧数 在判断连续执行时使用
    a_temp = 0
    b_temp = 0
    c_temp = 0
    d_temp = 0

    # 记录步骤执行的帧数  a是第一步 b是第二步 c是第三步 d是第四步
    a = 0
    b = 0
    c = 0
    d = 0

    # 标记位，标记是否开始检测
    # judge 为1 启动检测程序
    judge = dataInQt5Window2.get('judge')

    # 记录讯息
    message = ''

    # 记录一帧的四个步骤的匹配率
    match_ratio_list = [0.0, 0.0, 0.0, 0.0]

    # 抽帧保存路径
    save_dir = savePath + "/video2image"

    if os.path.exists(save_dir) is False:

        try:
            os.makedirs(save_dir)
        finally:
            pass

    # 抽帧
    cap = cv2.VideoCapture(videoPath)  # 生成读取视频对象
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取总帧数

    n = 1  # 计数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率

    i = 0
    timeF = int(fps)  # 视频帧计数间隔频率
    # 获取视频宽度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 获取视频高度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    photo_size = (frame_width, frame_height)
    videoWriter = cv2.VideoWriter(savePath + '/result.mp4',
                                  cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                  fps,
                                  photo_size)
    while cap.isOpened():
        ret, frame = cap.read()  # 按帧读取视频
        # 到视频结尾时终止
        if ret is False:
            break
        # 每隔timeF帧进行存储操作
        # if n % timeF == 0:
        if True:

            try:
                i += 1
                print(f'保存第 {i} 张图像')
                save_image_dir = os.path.join(save_dir, '%s.jpg' % i)
                print('save_image_dir: ', save_image_dir)
                cv2.imwrite(save_image_dir, frame)  # 保存视频帧图像
            finally:
                pass

            frame = r.search(save_image_dir).group(0)
            frame = frame[:-4]

            # match 路径
            resultPath4Match = os.path.join(matchPath, frame)

            frame = int(frame)
            img = cv2.imread(save_image_dir, 0)

            # 一切都得从第一个步骤开始
            temple1 = templesPath = os.path.join(templePath, temples_list[0])
            result = match.match(temple1, save_image_dir, resultPath4Match)
            # (1240, 494, 324, 373)

            # cropSize = [(1120, 730, 300, 400), (1750, 500, 500, 700)]  # x y w h
            cropSize = dataInQt5Window2.get('cropSize')
            cropPath = [savePath + '/crop.jpg',
                        savePath + '/crop2.jpg']

            img1 = cv2.imread(save_image_dir, -1)  # 保持原格式

            # 匹配率
            matchRatio1 = result[1]

            if judge == 0:
                if matchRatio1 < YU_ZHI:
                    x, y, w, h = cropSize[0]
                    cropImg = img1[y:y + h, x:x + w]
                    # cv2.imshow('1',cropImg)
                    # cv2.waitKey(0)

                    try:
                        cv2.imwrite(cropPath[0], cropImg)
                    finally:
                        pass
                    jieGuo, imgComplete = twiceDetect.detect(cropPath[0],
                                                             model_folder,
                                                             state, gpu)
                    # 手腕与手肘处于水平状态 且手腕在框内 判断执行第一步骤
                    try:
                        if Rule.one(jieGuo[0][4], jieGuo[0][7], jieGuo[0][3],
                                    jieGuo[0][6], WUCHA=WUCHA):
                            judge = 1
                    except TypeError:  # 遇到这个异常多半是没有检测到人体
                        pass
            else:
                for j in range(len(temples_list)):
                    templesPath = os.path.join(templePath, temples_list[j])
                    result = match.match(templesPath, save_image_dir,
                                         resultPath4Match)

                    # 第几步骤
                    temple = result[0]

                    # 匹配率
                    matchRatio = result[1]
                    match_ratio_list[temple] = matchRatio

                    # 绘制标注框
                    left_top = result[2]
                    right_below = result[3]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(matchRatio),
                                (left_top[0], left_top[1] - 20,),
                                font, 1, (255, 255, 0), 2)
                    cv2.rectangle(img, left_top, right_below, (255, 0,), 2)

                    # 匹配率小于阈值
                    # or jq < ZLX
                    if matchRatio < YU_ZHI:
                        # if matchRatio < YU_ZHI:
                        #     jq = 0  # 如果满足条件，刷新jq
                        try:
                            if temple <= 1:
                                x, y, w, h = cropSize[0]
                                cropImg = img1[y:y + h, x:x + w]

                                try:
                                    cv2.imwrite(cropPath[0], cropImg)
                                finally:
                                    pass
                                jieGuo, imgComplete = twiceDetect.detect(
                                    cropPath[0], model_folder, state, gpu)
                                img1[y:y + h, x:x + w] = imgComplete
                            else:
                                x, y, w, h = cropSize[1]
                                cropImg = img1[y:y + h, x:x + w]

                                try:
                                    cv2.imwrite(cropPath[0], cropImg)
                                finally:
                                    pass
                                jieGuo, imgComplete = twiceDetect.detect(
                                    cropPath[1], model_folder, state, gpu)
                                img1[y:y + h, x:x + w] = imgComplete
                            for jg in jieGuo:
                                if temple == 0:
                                    if Rule.one(jg, WUCHA=WUCHA):
                                        a += 1

                                elif temple == 1:
                                    if Rule.two(jg, WUCHA=WUCHA):
                                        b += 1

                                elif temple == 2:
                                    if Rule.three(jg, WUCHA=WUCHA):
                                        c += 1

                                elif temple == 3:
                                    if Rule.four(jg, WUCHA=WUCHA):
                                        d += 1
                                else:
                                    message += '出错！'
                        except TypeError:
                            pass

                        #  判断是否乱序
                        if a == 0 and (
                                b > LUAN_XU or c > LUAN_XU or d > LUAN_XU):
                            message = "乱序 第一个步骤未执行"
                            flag = 5
                            # break
                        elif b == 0 and (c > LUAN_XU or d > LUAN_XU):
                            message = "乱序 第二个步骤未执行"
                            flag = 6
                            # break
                        elif c == 0 and (d > LUAN_XU):
                            message = "乱序 第三个步骤未执行"
                            flag = 7
                            # break
                    else:
                        # 删除文件夹及其内容
                        shutil.rmtree(result[4])
                        if len(os.listdir(resultPath4Match)) == 0:
                            shutil.rmtree(resultPath4Match)
            img_path = images_dirs + "/" + str(frame) + ".jpg"

            try:
                cv2.imwrite(img_path, img1)
            finally:
                pass
            videoWriter.write(img1)

        # else:
        #     videoWriter.write(frame)
        # jq = jq + 1
        n = n + 1
        # 不加这一句windows会认为窗口死了，但是明明没运行到这里
        cv2.waitKey(1)  # 延时1ms
    shutil.rmtree(save_dir)
    shutil.rmtree(images_dirs)
    shutil.rmtree(matchPath)
    if len(message) == 0:
        if a == 0:
            message = '缺项 第一个步骤缺失'
            flag = 1
        elif b == 0:
            message = '缺项 第二个步骤缺失'
            flag = 2

        elif c == 0:
            message = '缺项 第三个步骤缺失'
            flag = 3

        elif d == 0:
            message = '缺项 第四个步骤缺失'
            flag = 4

        else:
            message = "流程正确执行"
            flag = 0
    videoWriter.release()
    cap.release()  # 释放视频对象

    message = message + " 耗时:" + time.strftime("%H:%M:%S", time.gmtime(
        time.perf_counter() - start))

    # save.save(frames_num=i, a=a, b=b, c=c, d=d, message=message, games_path=matchPath, yu_zhi=YU_ZHI, wu_cha=WUCHA,
    #           luan_xu=LUAN_XU)

    message = '第一个步骤%d,第二个步骤%d，第三个步骤%d，第四个步骤%d \n' % (
        a, b, c, d) + message
    return message, flag, frames


# flag中，0表示正确执行，1、2、3、4表示缺项，分别时1、2、3、4缺失，5，6，7则表示乱序，分别表示1、2、3未执行

def error(e, start, state):
    """
    错误处理函数

    :param e: 异常
    :param start: 开始时间
    :param state: 状态
    """
    usedtimeSec = time.perf_counter() - start
    state['start'] += usedtimeSec
    with open(state_file, 'wb') as f:
        pickle.dump(state, f)
    traceback.print_exc()
    raise SystemExit(e)


def sigHandler(signum, frame, start, state):
    """
    信号处理函数，用于捕获程序中断信号并退出程序

    :param signum: 信号编号
    :type signum: int
    :param frame: 信号帧
    :type frame: Any
    :param start: 开始时间
    :type start: float
    :param state: 状态
    :type state: dict
    """
    error(signum, start, state)


def run_process_video(state_p, state_exclude, state_num, state_list,
                      state_framesl,
                      state_name, judge_flag, model_folder,
                      gpu):
    """
    处理视频，返回处理结果

    :param state_p: 处理的状态
    :type state_p: list[str]
    :param state_exclude: 排除的状态
    :type state_exclude: list[str]
    :param state_num: 状态编号
    :type state_num: int
    :param state_list: 状态列表
    :type state_list: list[int]
    :param state_framesl: 帧数列表
    :type state_framesl: list[int]
    :param state_name: 状态名称
    :type state_name: list[str]
    :param judge_flag: 判断标志
    :type judge_flag: tuple[int, int]
    :param model_folder: 模型文件夹
    :type model_folder: str
    :param l: 阈值
    :type l: float
    :param gpu: 是否使用GPU
    :type gpu: bool
    """

    for p in [x for x in state_name if x not in state_p]:
        dirPath = videos + p
        videoNames = os.listdir(dirPath)
        state_p.append(p)
        for videoName in [x for x in videoNames if x not in state_exclude]:
            state_exclude.append(videoName)
            state_num += 1
            videoPath = os.path.join(dirPath, videoName)
            fps, flagt, framest = run(videoPath, model_folder, gpu)
            if judge_flag[0] <= flagt <= judge_flag[1]:
                state_list.append(flagt)
            state_framesl += framest


def run_main(start, state):
    """
    处理视频，返回处理结果

    :param start: 开始
    :param state: 状态
    """

    p1 = Process(target=run_process_video,
                 kwargs={'state_p': state['p'],
                         'state_exclude': state['rightVideoNames'],
                         'state_num': state['num0'],
                         'state_list': state['list0'],
                         'state_framesl': state['framesl1'],
                         'state_name': state['right'], 'judge_flag': (0, 0),
                         'model_folder': 'openpose/models', 'gpu': '0'})
    p2 = Process(target=run_process_video,
                 kwargs={'state_p': state['p'],
                         'state_exclude': state['lackVideomNames'],
                         'state_num': state['num1'],
                         'state_list': state['list1'],
                         'state_framesl': state['framesl2'],
                         'state_name': state['lack'], 'judge_flag': (1, 4),
                         'model_folder': 'openposeCNN/models', 'gpu': '0'})
    p3 = Process(target=run_process_video,
                 kwargs={'state_p': state['p'],
                         'state_exclude': state['outVideoNames'],
                         'state_num': state['num2'],
                         'state_list': state['list2'],
                         'state_framesl': state['framesl3'],
                         'state_name': state['out'], 'judge_flag': (5, 7),
                         'model_folder': 'models', 'gpu': '1'})

    p1.start()
    p1.join()
    p2.start()
    p2.join()
    p3.start()
    p3.join()
    if p1.exitcode != 0:
        # 子进程不是正常退出，处理异常情况并关闭主进程
        print(f"p1子进程异常退出，退出状态码为{p1.exitcode}")
        error(p1.exitcode, start, state)
        exit(1)
    else:
        usedtimeSec = time.perf_counter() - start + state['start']
        save.save4linux4one(state['num0'], state['list0'], usedtimeSec,
                            state['framesl1'], 'right')
        print(f"p1子进程正常退出，退出状态码为{p1.exitcode}")
    if p2.exitcode != 0:
        # 子进程不是正常退出，处理异常情况并关闭主进程
        print(f"p2子进程异常退出，退出状态码为{p2.exitcode}")
        error(p2.exitcode, start, state)
        exit(1)
    else:
        usedtimeSec = time.perf_counter() - start + state['start']
        save.save4linux4one(state['num1'], state['list1'], usedtimeSec,
                            state['framesl2'], 'lack')
        print(f"p2子进程正常退出，退出状态码为{p2.exitcode}")
    if p3.exitcode != 0:
        # 子进程不是正常退出，处理异常情况并关闭主进程
        print(f"p3子进程异常退出，退出状态码为{p3.exitcode}")
        error(p3.exitcode, start, state)
        exit(1)
    else:
        usedtimeSec = time.perf_counter() - start + state['start']
        save.save4linux4one(state['num2'], state['list2'], usedtimeSec,
                            state['framesl3'], 'out')
        print(f"p3子进程正常退出，退出状态码为{p3.exitcode}")


#
if __name__ == '__main__':
    with open('config.yml', 'r', encoding='utf-8') as y:
        dataInQt5Window2 = yaml.safe_load(y)
        # 定义保存状态的文件路径
    state_file = dataInQt5Window2.get('state_file')
    tasks = ['right', 'lack', 'outOForder']

    # 尝试加载上一次保存的状态

    try:
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
    except FileNotFoundError:
        # 如果找不到状态文件，则说明程序是第一次执行，初始化状态
        state = {
            'num0': 0,  # 已经检测的视频数
            'num1': 0,  # 已经检测的视频数
            'num2': 0,  # 已经检测的视频数
            'list0': [],  # 用于保存检测结果
            'list1': [],  # 用于保存检测结果
            'list2': [],  # 用于保存检测结果
            'framesl1': 0,  # 已经检测的帧数
            'framesl2': 0,  # 已经检测的帧数
            'framesl3': 0,  # 已经检测的帧数
            'rightVideoNames': [],  # 已处理的文件
            'right': ['1234/'],  # 需要处理的文件夹
            'lackVideomNames': [],  # 已处理的文件
            'lack': ['12/', '13/', '14/', '123/', '124/', '134/'],  # 需要处理的文件夹
            'outVideoNames': [],  # 已处理的文件
            'out': ['1423/', '1432/', '1342/', '1324/'],  # 需要处理的文件夹
            # 其他需要保存的状态
            'start': 0.0,  # 开始时间
            'p': [],  # 已处理的文件夹
        }
    start = time.perf_counter()
    try:
        sig_handler = functools.partial(sigHandler, start=start, state=state)
        signal.signal(signal.SIGABRT, sig_handler)
        signal.signal(signal.SIGINT, sig_handler)
        run_main(start, state)
    except Exception as e:
        error(e, start, state)

    os.remove(state_file)
    # 计时结束
    usedtimeSec = time.perf_counter() - start + state['start']

    # usedtime = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start))
    save.save4linux(state['num0'], state['num1'], state['num2'], state['list0'],
                    state['list1'], state['list2'],
                    usedtimeSec,
                    state['framesl1'] + state['framesl2'] + state['framesl3'])
