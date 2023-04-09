import concurrent
import os
import pickle
import re
import shutil
import sys
import time

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


def run(videoPath):
    #
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
    # print(temples_list)
    # 生成保存路径
    i = 1
    savePath = resultPath + "/" + str(i)
    while os.path.exists(savePath):
        i = i + 1
        savePath = resultPath + "/" + str(i)
    else:
        os.makedirs(savePath)

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
    os.makedirs(images_dirs)

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
        os.makedirs(save_dir)

    # 抽帧
    cap = cv2.VideoCapture(videoPath)  # 生成读取视频对象
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
    # print("视频总帧数:", frames)
    # print("视频帧率:", fps)
    # print("视频宽度:", width)
    # print("视频高度:", height)
    # print("视频时长:", frames / fps)

    # 保存视频信息
    with open(savePath + "/videoInfo.txt", 'w') as f:
        f.write("视频总帧数:" + str(frames) + "
    i = 1
    savePath = resultPath + "/" + str(i)
    while os.path.exists(savePath):
        i = i + 1
        savePath = resultPath + "/" + str(i)
    else:
        os.makedirs(savePath)

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
    os.makedirs(images_dirs)

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
        os.makedirs(save_dir)

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
    videoWriter = cv2.VideoWriter(savePath + '/result.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps,
                                  photo_size)
    while cap.isOpened():
        ret, frame = cap.read()  # 按帧读取视频
        # 到视频结尾时终止
        if ret is False:
            break
        # 每隔timeF帧进行存储操作
        # if n % timeF == 0:
        if True:
            i += 1
            print(f'保存第 {i} 张图像')
            save_image_dir = os.path.join(save_dir, '%s.jpg' % i)
            print('save_image_dir: ', save_image_dir)
            cv2.imwrite(save_image_dir, frame)  # 保存视频帧图像

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
                    cv2.imwrite(cropPath[0], cropImg)
                    jieGuo, imgComplete = twiceDetect.detect(cropPath[0])
                    # 手腕与手肘处于水平状态 且手腕在框内 判断执行第一步骤
                    try:
                        if Rule.one(jieGuo[0][4], jieGuo[0][7], jieGuo[0][3], jieGuo[0][6], WUCHA=WUCHA):
                            judge = 1
                    except TypeError:  # 遇到这个异常多半是没有检测到人体
                        pass
            else:
                for j in range(len(temples_list)):
                    templesPath = os.path.join(templePath, temples_list[j])
                    result = match.match(templesPath, save_image_dir, resultPath4Match)

                    # 第几步骤
                    temple = result[0]

                    # 匹配率
                    matchRatio = result[1]
                    match_ratio_list[temple] = matchRatio

                    # 绘制标注框
                    left_top = result[2]
                    right_below = result[3]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(matchRatio), (left_top[0], left_top[1] - 20,),
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
                                cv2.imwrite(cropPath[0], cropImg)
                                jieGuo, imgComplete = twiceDetect.detect(cropPath[0])
                                img1[y:y + h, x:x + w] = imgComplete
                            else:
                                x, y, w, h = cropSize[1]
                                cropImg = img1[y:y + h, x:x + w]
                                cv2.imwrite(cropPath[1], cropImg)
                                jieGuo, imgComplete = twiceDetect.detect(cropPath[1])
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
                        if a == 0 and (b > LUAN_XU or c > LUAN_XU or d > LUAN_XU):
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

            cv2.imwrite(img_path, img1)
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

    message = message + " 耗时:" + time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start))

    # save.save(frames_num=i, a=a, b=b, c=c, d=d, message=message, games_path=matchPath, yu_zhi=YU_ZHI, wu_cha=WUCHA,
    #           luan_xu=LUAN_XU)

    message = '第一个步骤%d,第二个步骤%d，第三个步骤%d，第四个步骤%d \n' % (a, b, c, d) + message
    return message, flag, frames


tasks = ['right', 'lack', 'outOForder']
# 定义保存状态的文件路径
state_file = 'state.pkl'
# 尝试加载上一次保存的状态
try:
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
except FileNotFoundError:
    # 如果找不到状态文件，则说明程序是第一次执行，初始化状态
    state = {
        'num0': 0,
        'num1': 0,
        'num2': 0,
        'list0': [],
        'list1': [],
        'list2': [],
        'framesl': 0,
        'rightVideoNames': [],
        'right': ['1234/'],
        'lackVideomNames': [],
        'lack': ['12/', '13/', '14/', '123/', '124/', '134/'],
        'outVideoNames': [],
        'out': ['1423/', '1432/', '1342/', '1324/'],
        # 其他需要保存的状态
        'start': 0.0,
        'p': [],
    }
start = time.perf_counter()


# flag中，0表示正确执行，1、2、3、4表示缺项，分别时1、2、3、4缺失，5，6，7则表示乱序，分别表示1、2、3未执行

# 处理正确列表中的视频
def process_videos(videos, state, executor, video_list, state_list, state_video_names, state_num, flag_min, flag_max):
    # 遍历视频列表
    for p in [x for x in video_list if x not in state['p']]:
        # 获取视频路径
        dirPath = videos + p
        # 获取视频名称列表
        videoNames = os.listdir(dirPath)
        # 将视频名称添加到已处理列表中
        state['p'].append(p)
        # 创建future列表
        futures = []
        # 遍历视频名称列表
        for videoName in [x for x in videoNames if x not in state_video_names]:
            # 将视频名称添加到已处理列表中
            state_video_names.append(videoName)
            # 帧数加1
            state['num0'] += 1
            # 获取视频路径
            videoPath = os.path.join(dirPath, videoName)
            # 提交任务
            futures.append(executor.submit(run, videoPath))
        # 遍历future列表
        for future in concurrent.futures.as_completed(futures):
            # 获取结果
            _, flagt, framest = future.result()
            # 判断flag是否在指定范围内
            if flag_min <= flagt <= flag_max:
                # 将flag添加到状态列表中
                state_list.append(flagt)
            # 帧数加上当前视频的帧数
            state['framesl'] += framest

try:
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # 处理正确列表中的视频
        process_videos(videos, state, executor, [x for x in state['right'] if x not in state['p']], state['list0'], state['rightVideoNames'], state['num0'], 0, 0)
        # 处理缺项列表中的视频
        process_videos(videos, state, executor, [x for x in state['lack'] if x not in state['p']], state['list1'], state['lackVideomNames'], state['num1'], 1, 4)
        # 处理乱序列表中的视频
        process_videos(videos, state, executor, [x for x in state['outOForder'] if x not in state['p']], state['list2'], state['outVideoNames'], state['num2'], 5, 7)

# 捕获异常
except:
    # 计算已用时间
    usedtimeSec = time.perf_counter() - start
    # 更新开始时间
    state['start'] += usedtimeSec
    # 将状态保存到文件中
    with open(state_file, 'wb') as f:
        pickle.dump(state, f)
    # 输出错误信息
    print('error find,interrupt')
    # 退出程序
    sys.exit(0)


# 所有任务执行完成后，删除状态文件
os.remove(state_file)
# 计时结束
usedtimeSec = time.perf_counter() - start + state['start']

# usedtime = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start))
save.save4linux(state['num0'], state['num1'], state['num2'], state['list0'], state['list1'], state['list2'],
                usedtimeSec, state['framesl'])
