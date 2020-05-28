import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np


def shape_to_np(shape):
    coords = np.zeros((68, 2))
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


class FaceDetection:
    def __init__(self):
        self.predictor_path = "dfm/shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def get_faces(self, img):
        return self.detector(img, 1)

    def get_points(self, image):
        faces = self.get_faces(image)
        points = [self.predictor(image, f) for f in faces]
        return [shape_to_np(p) for p in points]

    def img_correct(self, img):
        r, g, b = img.T[0], img.T[1], img.T[2]
        return np.array([b, g, r]).T

    def demo(self):
        image = cv2.imread("dfm/arny.jpg")
        points = self.get_points(image)

        plt.imshow(self.img_correct(image))
        [plt.scatter(p[:, 0], p[:, 1], c="r", s=5, marker="x") for p in points]
        plt.show()


if __name__ == '__main__':
    fd = FaceDetection()
    fd.demo()
