# -*- coding:utf-8 -*-
import time


def time2stamp(tss):
    """
    秒级别的时间戳转换
    :param tss: str.eg.tss = "2019-10-31 23:40:00"
    :return: int.eg.1572478460
    """
    timeArray = time.strptime(tss, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


def stamp2time(stamp):
    """
    秒级别的时间戳转换
    :param stamp: int.eg.1572478460
    :return: str.eg.tss = "2019-10-31 23:40:00"
    """
    # stamp = 1572529200
    timeArray = time.localtime(stamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def test_time2stamp():
    tss1 = "2019-10-31 21:40:00"
    tss2 = "2019-10-31 23:40:00"
    stamp1 = time2stamp(tss1)
    stamp2 = time2stamp(tss2)
    print(stamp1, stamp2)
    print((stamp2-stamp1))


def test_stamp2time():
    stamp1 = 1572529200
    stamp2 = 1572536400
    t1 = stamp2time(stamp1)
    t2 = stamp2time(stamp2)
    print(t1, t2)


if __name__ == '__main__':
    test_stamp2time()
    test_time2stamp()