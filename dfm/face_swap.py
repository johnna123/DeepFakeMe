from . import FaceDetection
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


class FaceSwapper:
    def __init__(self):
        self.fd = FaceDetection()

    def get_tri(self, points):
        return tri.Triangulation(points[:, 0], points[:, 1])

    def preprocess_img(self, img):
        points = self.fd.get_points(img)[0]
        x1 = np.float(min(points.max(axis=0)[0], img.shape[1]))
        y1 = np.float(min(points.max(axis=0)[1], img.shape[0]))
        x0 = np.float(max(points.min(axis=0)[0], 0))
        y0 = np.float(max(points.min(axis=0)[1], 0))
        points[:, 0] -= x0
        points[:, 1] -= y0
        img = img[int(y0):int(y1), int(x0):int(x1)]
        return img, points.astype(np.int32), (x0, y0, x1, y1)

    def face_swap(self, target_path, source_path):
        target_image = cv2.imread(target_path)
        source_image = cv2.imread(source_path)
        target, target_points, target_rect = self.preprocess_img(target_image)
        source, source_points, source_rect = self.preprocess_img(source_image)
        return self.core(source, source_points, target, target_image, target_points, target_rect)

    def core(self, source, source_points, target, target_image, target_points, target_rect):
        xt, yt, ct = target.shape
        mask_shape = (xt, yt, ct)
        target_triangs = self.get_tri(target_points)
        source_triangs = self.get_tri(source_points)
        t = [np.array([[target_triangs.x[p], target_triangs.y[p]] for p in triang]).astype(np.float32) for triang in
             target_triangs.triangles]
        s = [np.array([[source_triangs.x[p], source_triangs.y[p]] for p in triang]).astype(np.float32) for triang in
             target_triangs.triangles]
        s2ttrans = [cv2.getAffineTransform(ss, tt) for ss, tt in zip(s, t)]
        out = np.zeros(mask_shape, dtype=np.uint8)
        for triangle, m in zip(target_triangs.triangles, s2ttrans):
            mask = np.zeros(mask_shape, dtype=np.uint8)
            roi_corners = np.array([[target_triangs.x[p], target_triangs.y[p]] for p in triangle],
                                   dtype=np.int32)
            ignore_mask_color = (255,) * ct
            cv2.fillPoly(mask, [roi_corners], ignore_mask_color)

            dst = cv2.warpAffine(source, m, (mask_shape[1], mask_shape[0]), cv2.INTER_LINEAR, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_TRANSPARENT)

            masked_image = cv2.bitwise_and(dst, mask)

            # out += masked_image

            out = cv2.bitwise_or(out, masked_image)
        aux = target_image.copy()
        aux[int(target_rect[1]):int(target_rect[1] + out.shape[0]),
        int(target_rect[0]):int(target_rect[0] + out.shape[1])] = out
        mask = np.zeros(target_image.shape, dtype=np.uint8)
        hull_points = cv2.convexHull(target_points.astype(np.int32))
        hull_points = hull_points.squeeze()
        hull_points[:, 0] += int(target_rect[0])
        hull_points[:, 1] += int(target_rect[1])
        cv2.fillConvexPoly(mask, np.int32(hull_points), (255, 255, 255), 16, 0)
        x2 = np.float(min(hull_points.max(axis=0)[0], target_image.shape[1]))
        y2 = np.float(min(hull_points.max(axis=0)[1], target_image.shape[0]))
        x1 = np.float(max(hull_points.min(axis=0)[0], 0))
        y1 = np.float(max(hull_points.min(axis=0)[1], 0))
        center = (int(np.floor((x1 + x2) / 2)), int(np.floor((y1 + y2) / 2)))
        output = cv2.seamlessClone(aux, target_image, mask, center, cv2.NORMAL_CLONE)
        # return self.fd.img_correct(output)
        return output


if __name__ == '__main__':
    FS = FaceSwapper()
    # a,b,c,=FS.preprocess_img(cv2.imread("target_data/frame_2711.jpg"))
    # plt.imshow(a)
    # plt.scatter(b[:,0],b[:,1],c="r")
    # plt.show()
    plt.imshow(FS.face_swap("images/targs/temp.jpg", "images/sourcs/ga6.jpg"))
    plt.show()
