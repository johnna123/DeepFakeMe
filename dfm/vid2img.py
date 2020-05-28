import cv2
import argparse


def vid2img(path, video_name, start, stop, skip_frames=False):
    video = cv2.VideoCapture(path + video_name)
    fps = video.get(5)
    fstart = start * fps
    fstop = stop * fps
    count = 0
    skip_count = 0
    while True:
        data_exist, data = video.read()
        if data_exist and count <= fstop:
            count += 1
            skip_count += 1
            if skip_frames and skip_count > fps:
                skip_count = 0
                if fstart <= count:
                    cv2.imwrite(path + "frame_{}.jpg".format(count), data)
            if not skip_frames:
                skip_count = 0
                if fstart <= count:
                    cv2.imwrite(path + "frame_{}.jpg".format(count), data)
        else:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("video_name", type=str)
    parser.add_argument("-start", "--start", type=int, default=0)
    parser.add_argument("-stop", "--stop", type=int, default=float("inf"))
    args = parser.parse_args()

    vid2img(args.path, args.video_name, args.start, args.stop)
