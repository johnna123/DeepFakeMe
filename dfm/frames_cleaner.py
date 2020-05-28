import cv2
import os
from . import FaceDetection
import argparse

fd = FaceDetection()


def cleaner(data_dir):
    files = os.listdir(data_dir)
    for file in files:
        print(file)
        if ".jpg" in file:
            if len(fd.get_faces(cv2.imread(data_dir + file))) != 1:
                os.remove(data_dir + file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    cleaner(args.path)
