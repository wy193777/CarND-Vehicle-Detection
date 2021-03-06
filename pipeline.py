from collections import deque
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import svc
import pickle
from pathlib import Path
from moviepy.editor import VideoFileClip


class Pipeline(object):

    def __init__(self, svc, scaler):
        self.svc = svc
        self.scaler = scaler
        self.history = deque(maxlen=8)

    # Define a single function that can extract features using hog sub-sampling
    # and make predictions.
    def find_cars(
        self, img, ystart=400, ystop=650, scale=1.5, orient=9,
        pix_per_cell=8, cell_per_block=2, vir=False
    ):

        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = svc.convert_color(
            img_tosearch, cspace='YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(
                ctrans_tosearch,
                (np.int(imshape[1] / scale), np.int(imshape[0] / scale))
            )

        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        # Instead of overlap, define how many cells to step
        cells_per_step = 2
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hogs = list(
            svc.get_hog_features(
                ctrans_tosearch[:, :, idx], orient, pix_per_cell,
                cell_per_block,
                feature_vec=False
            )
            for idx in range(3)
        )
        bbox_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_features = np.hstack([
                    hog[
                        ypos:ypos + nblocks_per_window,
                        xpos:xpos + nblocks_per_window
                    ].ravel()
                    for hog in hogs
                ])

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(
                    ctrans_tosearch[ytop:ytop + window, xleft:xleft + window],
                    (64, 64)
                )

                features = hog_features
                test_features = self.scaler.transform(features.reshape(1, -1))
                test_prediction = self.svc.predict(test_features)
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)

                    bbox_list.append(
                        (
                            (xbox_left, ytop_draw + ystart),
                            (xbox_left + win_draw,
                                ytop_draw + win_draw + ystart)
                        )
                    )
                if vir:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(
                        draw_img,
                        (xbox_left, ytop_draw + ystart),
                        (xbox_left + win_draw,
                            ytop_draw + win_draw + ystart),
                        (0, 0, 127), 4)

        return draw_img if vir else bbox_list

    def find_cars_scale(
        self,
        img,
        scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
        vir=False
    ):
        result = []
        if vir:
            for sc in scales:
                result.append(self.find_cars(img, scale=sc, vir=vir))
        else:
            for sc in scales:
                result.extend(self.find_cars(img, scale=sc, vir=vir))
        return result

    def process(self, img):
        bbox_list = self.find_cars_scale(img)
        heatmap = svc.add_heat(bbox_list, threshold=3)
        self.history.append(heatmap)
        heatmap_sum = np.mean(np.array(self.history), axis=0)
        out_img = svc.draw_labeled_bboxes(img, heatmap_sum)
        return out_img


def init_pipeline():
    svc_scaler = Path("data.p")
    if svc_scaler.is_file():
        result = pickle.load(open("data.p", 'rb'))
    else:
        result = svc.generate_svc()
        pickle.dump(result, open("data.p", 'wb'))
    # import ipdb; ipdb.set_trace()
    return Pipeline(*result), svc_scaler


def process_video(video_path, output_path):
    pipeline, svc_scaler = init_pipeline()
    clip1 = VideoFileClip(video_path)
    output_video = clip1.fl_image(pipeline.process)
    output_video.write_videofile(output_path, audio=False)


def generate_images():
    pipeline, svc_scaler = init_pipeline()
    for path in glob.glob('test_images/*.jpg'):
        img = misc.imread(path)
        bbox_list = pipeline.find_cars_scale(img)
        heatmap = svc.add_heat(bbox_list, threshold=2)
        misc.imsave(
            'test_images_output/heatmap_' + Path(path).name,
            np.clip(heatmap, 0, 255)
        )
        out_img = svc.draw_labeled_bboxes(img, heatmap)
        misc.imsave('test_images_output/label_' + Path(path).name, out_img)


if __name__ == "__main__":
    # process_video('test_video.mp4', 'output_test_video.mp4')
    process_video('project_video.mp4', 'output_project_video.mp4')

    # generate_images()
