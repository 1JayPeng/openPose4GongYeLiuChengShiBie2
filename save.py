import os.path

import pandas as pd
#  为了方便收集每个有效数据的result.csv，这里参数传递 @games_path 定义为所有图片的检测结果文件夹，真正保存文件夹使用该路径的上级目录
from numpy import mean


def save4linux(num0, num1, num2, list0, list1, list2, time, Frames):
    pass
    pro0 = len(list0) / num0
    pro1 = len(list1) / num1
    pro2 = len(list2) / num2
    sImages = time / Frames

    df = pd.DataFrame(
        {'true': pro0, 'lack': pro1, 'out': pro2, 'sImages': sImages,
         'time': time, 'totalFrames': Frames,
         'num_true': num0,
         'num_lack': num1,
         'num_out': num2,

         }
    )
    path = '../result'
    df.to_csv(path + "/result.csv", sep=',', encoding="utf-8-sig")


def save4linux4one(num, list, time, Frames, judge):
    pass
    pro = len(list) / num
    sImages = time / Frames

    df = pd.DataFrame(
        {judge: pro, 'sImages': sImages, 'time': time, 'totalFrames': Frames,
         'num_true': num,
         }
    )
    path = '../result'
    df.to_csv(path + "/result_" + judge + ".csv", sep=',', encoding="utf-8-sig")


def save(frames_num, a, b, c, d, message, games_path, yu_zhi, wu_cha, luan_xu):
    # a = detect_list[0]
    # b = detect_list[1]
    # c = detect_list[2]
    # d = detect_list[3]
    # message = detect_list[4]
    # games_path = detect_list[5]
    frame_num = [a, b, c, d]
    max_val_list = [[],
                    [],
                    [],
                    []]
    print(type(max_val_list))
    max_val_ave_list = []
    print(type(max_val_ave_list))
    save_path = os.path.dirname(games_path)

    games_list = os.listdir(games_path)  # 图片序号
    for i in games_list:  # 图片序号
        game_path = os.path.join(games_path, i)
        steps_list = os.listdir(game_path)
        for j in steps_list:
            step_path = os.path.join(game_path, j)
            csv_path = os.path.join(step_path, 'result.csv')
            reader = pd.read_csv(csv_path, sep=',', header=0)
            bu_zhou = reader.at[0, '步骤']
            loc_val = reader.at[0, 'loc_val']
            print(reader)
            # print(bu_zhou)
            # print(max_val)
            max_val_list[bu_zhou].append(loc_val)
            # max_val_list[row['步骤']].append(row['max_val'])
    for i in range(4):
        # print (mean(max_val_list[i]))
        if max_val_list[i] is None:
            continue
        max_val_ave = float(mean(max_val_list[i]))
        max_val_ave_list.append(max_val_ave)
    df = pd.DataFrame(
        {
            "阈值": yu_zhi, "界定值": wu_cha, "容错值": luan_xu,
            "有效帧数": frame_num, "平均最大值": max_val_ave_list,
            "视频总帧数": frames_num, "备注": message
        }, index=["步骤1", "步骤2", "步骤3", "步骤4"]
    )

    df.to_csv(save_path + "\\result.csv", sep=',', encoding="utf-8-sig")
