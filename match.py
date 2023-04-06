import os
import time

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

# 读取yaml配置文件
with open('config.yml', 'r', encoding='utf-8') as y:
    dataInMatch = yaml.safe_load(y)


def quHeiBian(imagesPath):
    NUM = 40  # 阈值
    width = 240
    height = 160

    gray = cv2.imread(imagesPath)  # 导入图片
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)  # 转换为灰度图像

    nrow = gray.shape[0]  # 获取图片尺寸
    ncol = gray.shape[1]
    # print(ncol, nrow)
    rowc = gray[:, int(1 / 2 * nrow)]  # 无法区分黑色区域超过一半的情况
    colc = gray[int(1 / 2 * ncol), :]

    rowflag = np.argwhere(rowc > NUM)
    colflag = np.argwhere(colc > NUM)

    left, bottom, right, top = rowflag[0, 0], colflag[-1, 0], rowflag[-1, 0], colflag[0, 0]
    result = gray[left:right, top:bottom]

    # result = cv2.resize(result, (width, height))
    # print(result.shape[1], result.shape[0])

    # cv2.imshow('name', result)  # 效果展示
    # cv2.imwrite(r"E:\AI\result\1.jpg", result)
    # cv2.waitKey(0)
    return result


def match(temple_path, game_path, save_path, isShow=False):
    start = time.perf_counter()
    # print(temple_path[-5:-4])
    # 读取待检测图像
    img = cv2.imread(game_path, 0)
    # 对图像进行去黑边处理，防止干扰检测
    # img = quHeiBian(game_path)
    # 读取模板图像
    temple = cv2.imread(temple_path, 0)
    # print(temple.shape[1], temple.shape[0])  # 宽高
    whichTemples = temple_path[-5:-4]
    # LOC_LIST = dataInMatch.get('LOC_LIST')
    # 使用标准相关系数进行匹配
    # MATCH_WAY = cv2.TM_CCOEFF_NORMED
    # MATCH_WAY = cv2.TM_CCOEFF
    MATCH_WAY = cv2.TM_CCOEFF_NORMED  # 标准相关匹配

    # 创建保存目录
    i = 1
    savePath = save_path + "/" + str(i)
    while os.path.exists(savePath):
        i = i + 1
        savePath = save_path + "/" + str(i)
    else:
        os.makedirs(savePath)
    if isShow:
        # 显示灰度处理后的待检测图像
        cv2.namedWindow('sample', 0)
        cv2.resizeWindow('sample', 400, 600)
        cv2.imshow('sample', img)

        # 显示灰度处理后的模板图像
        cv2.namedWindow('target', 0)
        cv2.resizeWindow('target', 400, 600)
        cv2.imshow('target', temple)
    # 高 宽
    th, tw = temple.shape[:2]
    # print(th, tw)

    result = cv2.matchTemplate(img, temple, MATCH_WAY)
    if isShow:
        # result为匹配结果矩阵
        # print(result)
        # print(result[585][837])
        # result = result.T
        # print(result)
        # print(result[837][585])
        # TM_CCOEFF_NORMED方法处理后的结果图像
        cv2.namedWindow('match_r', 0)
        cv2.resizeWindow('match_r', 400, 600)
    picture = Image.fromarray(result * 255)

    picture = picture.convert("L")
    if isShow:
        # # 显示窗口
        cv2.imshow('match_r', result)

    # 使用函数minMaxLoc,确定匹配结果矩阵的最大值和最小值(val)，以及它们的位置(loc)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if isShow:
        print('min_val ,max_val :' + str(min_val), max_val)
        print('min_loc ,max_loc :', min_loc, max_loc)
    # 此处选取最大值的位置,为矩形的左上角#
    # 为避免干扰，不再选取最大值位置，转而对检测图像进行遮罩，只取模板图像对应点的值
    whichTemples = int(whichTemples) - 1
    if isShow:
        print('whichTemples:'+str(whichTemples))
        # print('LOC_LIST[whichTemples]:'+str(LOC_LIST[whichTemples]))
    # tl = LOC_LIST[whichTemples]
    tl = max_loc  # 宽 高
    # 获取矩形的右下角
    br = (tl[0] + tw, tl[1] + th)

    # # 设置显示窗口
    if isShow:
        cv2.namedWindow('match', 0)
        cv2.resizeWindow('match', 400, 600)
        cv2.namedWindow('match', 0)

    # 保存图片
    # cv2.imwrite(savePath + '\sample.jpg', img)
    # cv2.imwrite(savePath + '\\target.jpg', temple)
    picture.save(savePath + '/match_r.jpg')
    # cv2.imwrite(savePath + '\match_r.jpg', picture)
    # 绘制矩形框
    cv2.rectangle(img, tl, br, (255, 0,), 2)  # 参数：图像 矩形的左上角 矩形的右下角 色彩 宽度
    cv2.imwrite(savePath + '/match.jpg', img)
    if isShow:
        # # 显示窗口
        cv2.imshow('match', img)
    cv2.waitKey(0)
    result = result.T
    if isShow:
        print(result[tl[0], tl[1]])
        print(tl)
    # 保存结果以及路径信息
    frame = pd.DataFrame(
        {"loc_val": result[tl[0], tl[1]], "loc/(height,width)": (tl,), "image_path": game_path,
         "temple_path": temple_path, "匹配方式": MATCH_WAY, "步骤": whichTemples, "耗时": str(time.perf_counter() - start)})
    frame.to_csv(savePath + "/result.csv", sep=',', encoding="utf-8-sig")
    # 结束
    if isShow:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return whichTemples, result[tl[0], tl[1]], tl, br, savePath
