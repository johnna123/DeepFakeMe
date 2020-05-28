import joblib
from . import FaceSwapper
import cv2
import math
import numpy as np
import argparse

swapper = FaceSwapper()


def mse(v1, v2):
    return ((v1 - v2) ** 2).sum() / 68


def rel_dist(a, b, c, px, py):
    return (a * px + b * py + c) / (math.sqrt(a ** 2 + b ** 2))


def pos_vect(x1, x2, y1, y2, px, py):
    ah = y1 - y2
    bh = x2 - x1
    ch = y2 * x1 - y1 * x2
    av = x2 - x1
    bv = y2 - y1
    mx = (x2 + x1) / 2
    my = (y2 + y1) / 2
    cv = -(y1 * y2) - (x1 * x2) + (mx ** 2) + (my ** 2)  # mx my?
    disth = rel_dist(ah, bh, ch, px, py)
    distv = rel_dist(av, bv, cv, px, py)
    return np.array([disth, distv])


def best_swapper(source_path, target_path, processed_path):
    source_data = joblib.load(source_path + "data.pkl")
    target_data = joblib.load(target_path + "data.pkl")

    s_keys = list(source_data.keys())
    t_keys = list(target_data.keys())

    s_points = [source_data[k][1] for k in s_keys]
    t_points = [target_data[k][1] for k in t_keys]

    s_vects = [pos_vect(p[36, 0], p[45, 0], p[36, 1], p[45, 1], p[33, 0], p[33, 1]) for p in s_points]
    t_vects = [pos_vect(p[36, 0], p[45, 0], p[36, 1], p[45, 1], p[33, 0], p[33, 1]) for p in t_points]

    relations = []
    for t, k in zip(t_vects, t_keys):
        dists = [mse(t, s) for s in s_vects]
        relations.append([k, s_keys[dists.index(min(dists))]])

    sources = [source_data[k[1]][0] for k in relations]
    sps = [source_data[k[1]][1] for k in relations]
    targets = [target_data[k[0]][0] for k in relations]
    target_imgs = [cv2.imread(k[0]) for k in relations]
    tps = [target_data[k[0]][1] for k in relations]
    t_rects = [target_data[k[0]][2] for k in relations]

    # processed = [swapper.core(s, sp, t, ti, tp, tr) for s, sp, t, ti, tp, tr in
    #              zip(sources, s_points, targets, target_imgs, t_points, t_rects)]
    processed = []
    i = 0
    for s, sp, t, ti, tp, tr in zip(sources, sps, targets, target_imgs, tps, t_rects):
        print(i, t_keys[i])
        i += 1
        processed.append(swapper.core(s, sp, t, ti, tp, tr))

    [cv2.imwrite(processed_path + k.split("/")[1], p) for p, k in zip(processed, t_keys)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", type=str, default="source_data/")
    parser.add_argument("-t", "--target_path", type=str, default="target_data/")
    parser.add_argument("-p", "--processed_path", type=str, default="processed_data/")
    args = parser.parse_args()

    best_swapper(args.source_path, args.target_path, args.processed_path)
