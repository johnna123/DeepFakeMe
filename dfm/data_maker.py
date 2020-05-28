import cv2
import os
from . import FaceSwapper
import joblib
import argparse

swapper = FaceSwapper()


def create_data(data_dir):
    data = {}
    files = os.listdir(data_dir)
    for file in files:
        if ".jpg" in file:
            data[data_dir + file] = swapper.preprocess_img(cv2.imread(data_dir + file))
    joblib.dump(data, data_dir + "data.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    create_data(args.path)
