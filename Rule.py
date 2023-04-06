# 规则库文件
import yaml


def absSub(a, b, NUM=7):
    """求两个数的绝对差值，并判断是否小于NUM"""
    if abs(a - b) < NUM:
        return True
    else:
        return False


def inMiddle(left, mid, right):
    """比较三个值的大小，如果mid大于left小于right 返回真"""
    if left < mid < right:
        return True
    else:
        return False


# 读取yaml配置文件
with open('config.yml', 'r', encoding='utf-8') as y:
    dataInRule = yaml.safe_load(y)


def one(node, WUCHA=7):
    """
    这是步骤1的规则库 该规则是通过判断手和肘是否在一个大致的水平线上以及手是否到达指定位置来判断是否执行步骤1
    :type leftHand:array
    :param WUCHA: 浮动大小
    :param leftHand:左手腕节点 这些节点是一个数组，形式为[x,y.truth]
    :param rightHand:右手腕节点
    :param leftElbow:左手肘节点
    :param rightElbow:右手肘节点
    :return:boolean
    """
    recLoc = dataInRule.get('recLoc4one')
    leftHand = node[4]
    rightHand = node[7]
    leftElbow = node[3]
    rightElbow = node[6]
    neck = node[1]
    waist = node[8]
    if (  # 取y值就是判断手和肘是否在一个大致的水平线上
            absSub(leftHand[1], leftElbow[1], WUCHA) or
            absSub(rightHand[1], rightElbow[1], WUCHA)
    ) and \
            (
                    (inMiddle(recLoc[0], leftHand[0], recLoc[2]) and
                     inMiddle(recLoc[1], leftHand[1], recLoc[3])
                    ) or
                    (inMiddle(recLoc[0], rightHand[0], recLoc[2]) and
                     inMiddle(recLoc[1], rightHand[1], recLoc[3])
                    )
            ) and (absSub(neck[0], waist[0], WUCHA - 4)):
        return True
    else:
        return False


def two(node, WUCHA=7):
    """
    这是步骤2的规则库 同过判断手是否在一个大致的垂直线上以及手是否到达指定位置
    todo: 容易造成误差，与行走时的动作形态太过相似，在执行步骤1时容易误判，步骤3，4与步骤1、2同理，同样存在误判问题
    """
    recLoc = dataInRule.get('recLoc4two')
    leftHand = node[4]
    rightHand = node[7]
    leftElbow = node[3]
    rightElbow = node[6]
    neck = node[1]
    waist = node[8]
    if (  # 取x值就是判断手和肘是否在一个大致的竖直线上
            absSub(leftHand[0], leftElbow[0], WUCHA) or
            absSub(rightHand[0], rightElbow[0], WUCHA)
    ) and \
            (
                    (inMiddle(recLoc[0], leftHand[0], recLoc[2]) and
                     inMiddle(recLoc[1], leftHand[1], recLoc[3])
                    ) or
                    (inMiddle(recLoc[0], rightHand[0], recLoc[2]) and
                     inMiddle(recLoc[1], rightHand[1], recLoc[3])
                    )

            ) and (
            # 添加规则:腰部是否弯曲
            absSub(neck[0], waist[0], WUCHA + 10) is False):

        return True
    else:
        return False


def three(node, WUCHA=7):
    """

    """
    recLoc = dataInRule.get('recLoc4three')
    leftHand = node[4]
    rightHand = node[7]
    leftElbow = node[3]
    rightElbow = node[6]
    neck = node[1]
    waist = node[8]
    if (  # 取y值就是判断手和肘是否在一个大致的水平线上
            absSub(leftHand[1], leftElbow[1], WUCHA) or
            absSub(rightHand[1], rightElbow[1], WUCHA)
    ) and \
            (
                    (inMiddle(recLoc[0], leftHand[0], recLoc[2]) and
                     inMiddle(recLoc[1], leftHand[1], recLoc[3])
                    ) or
                    (inMiddle(recLoc[0], rightHand[0], recLoc[2]) and
                     inMiddle(recLoc[1], rightHand[1], recLoc[3])
                    )
            ) and (absSub(neck[0], waist[0], WUCHA - 4)):
        return True
    else:
        return False


def four(node, WUCHA=7):
    recLoc = dataInRule.get('recLoc4four')
    leftHand = node[4]
    rightHand = node[7]
    leftElbow = node[3]
    rightElbow = node[6]
    neck = node[1]
    waist = node[8]
    if (  # 取x值就是判断手和肘是否在一个大致的竖直线上
            absSub(leftHand[0], leftElbow[0], WUCHA) or
            absSub(rightHand[0], rightElbow[0], WUCHA)
    ) and \
            (
                    (inMiddle(recLoc[0], leftHand[0], recLoc[2]) and
                     inMiddle(recLoc[1], leftHand[1], recLoc[3])
                    ) or
                    (inMiddle(recLoc[0], rightHand[0], recLoc[2]) and
                     inMiddle(recLoc[1], rightHand[1], recLoc[3])
                    )
            ) and (  # 添加规则:腰部是否弯曲
            absSub(neck[0], waist[0], WUCHA + 10) is False):
        return True
    else:
        return False
