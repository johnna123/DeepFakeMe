import cv2
import os
import argparse


def vid_mod(target_path, processed_path, video_name, output_name):
    video = cv2.VideoCapture(target_path + video_name)
    fps = video.get(5)
    heigh = video.get(4)
    width = video.get(3)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path + output_name, fourcc, fps, (int(width), int(heigh)))
    count = 0
    modded = set(os.listdir(processed_path))
    while True:
        data_exist, data = video.read()
        if data_exist:
            count += 1
            if modded.intersection({"frame_{}.jpg".format(count)}):
                out.write(cv2.imread(processed_path + "frame_{}.jpg".format(count)))
            else:
                out.write(data)
        else:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target_path", type=str, default="target_data/")
    parser.add_argument("-p", "--processed_path", type=str, default="processed_data/")
    parser.add_argument("-v", "--video_name", type=str, default="target_vid.mp4")
    parser.add_argument("-o", "--output_name", type=str, default="modded.mp4")
    args = parser.parse_args()

    vid_mod(args.target_path, args.processed_path, args.video_name, args.output_name)
