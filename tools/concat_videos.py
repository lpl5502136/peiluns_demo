# -*- coding:utf-8 -*-
import os


def get_video_list(path):
    file_list = sorted(os.listdir(path))
    for file in file_list:
        print(file.split("/")[-1])
    return [file_name for file_name in file_list if file_name.find('.mp4') != -1]


def concat_video(path, video_list, out_file_name):
    command_string = 'ffmpeg -i "concat:'
    for video_name in video_list:
        video_name_absolute = os.path.join(path, video_name)
        command_string = command_string + video_name_absolute + '|'
    command_string = command_string[:-1] + '" -c copy -bsf:a aac_adtstoasc -movflags +faststart ' + os.path.join(path, out_file_name)
    os.system(command_string)


def concat_video_with_tmp(path, video_list, out_file_name):
    file_list_name = os.path.join(path, 'file_list.txt')
    with open(file_list_name, 'w') as f:
        for video_name in video_list:
            f.write("	file '{}'\n".format(video_name))
    os.system('ffmpeg -f concat -safe 0 -i {} -c copy {}'.format(file_list_name, out_file_name))
    # os.remove(file_list_name)


# def concat_all_sub_floder():
#     sub_path_list = [os.path.join(inner_path, sub_path)
#                      for sub_path in os.listdir(inner_path) if sub_path.find('2019') != -1]
#     for sub_path in sub_path_list:
#         video_list = get_video_list(sub_path)
#         concat_video_with_tmp(sub_path, video_list, sub_path + '.mp4')


def concat_all_floder_for_grab(inner_path):
    video_list = get_video_list(inner_path)
    concat_video_with_tmp(inner_path, video_list, inner_path + '.mp4')


def concat_all_floder(inner_path):
    video_list = get_video_list(inner_path)
    concat_video_with_tmp(inner_path, video_list, inner_path + '.mp4')


def main():
    inner_path = '../demo_cut'
    concat_all_floder(inner_path)

    # input = "../demo_cut"
    # sub_dirs = os.listdir(input)
    # for sub_dir in sub_dirs:
    #     sub_dir = os.path.join(input, sub_dir)
    #     # print(sub_dir)
    #     concat_all_floder_for_grab(sub_dir)


if __name__ == '__main__':
    main()
