# -*- coding: utf-8 -*-
# Author: lipeilun
import os
import time
import cv2
import numpy as np
import shutil
import glob


def flip_video(vdo_path, out_path):
    """
    水平翻转视频，读视频，写视频，图像水平翻转，获取视频信息
    :param vdo_path:
    :param out_path:
    :return: 读视频
    """
    out_fps = 25

    # 获取视频信息
    cap = cv2.VideoCapture(vdo_path)
    all_frames = int(cap.get(7))
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5)
    print(w, h, fps)

    # 写视频
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    output_movie = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))

    # 逐帧读视频
    for i in range(0, all_frames, 1):
        print(i)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print("not ret")
            continue

        # 图像水平翻转
        image = cv2.flip(frame, 1)

        output_movie.write(image)
    output_movie.release()


def test_flip_video():
    input = "./0.mp4"
    out = "./0flip.mp4"
    flip_video(input, out)


if __name__ == '__main__':
    test_flip_video()