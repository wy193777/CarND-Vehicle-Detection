import matplotlib.image as mpimg
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


class Pipeline(object):

    def __init__(self, svc, scaler):
        self.svc = svc
        self.scaler = scaler

    # Define a single function that can extract features using hog sub-sampling
    # and make predictions.
    def find_cars(
        self,
        img, ystart=400, ystop=650, scale=1.5, orient=9,
        pix_per_cell=8,
        cell_per_block=2, spatial_size=(32, 32), hist_bins=32
    ):

        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = svc.convert_color(
            img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(
                ctrans_tosearch,
                (np.int(imshape[1] / scale), np.int(imshape[0] / scale))
            )

        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
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

                # Get color features
                spatial_features = svc.bin_spatial(subimg, size=spatial_size)
                hist_features = svc.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                # features = np.hstack(
                #     (spatial_features, hist_features, hog_features)
                # )
                features = hog_features
                test_features = self.scaler.transform(features.reshape(1, -1))

                # test_features = (
                #     X_scaler
                #     .transform(np.hstack((shape_feat, hist_feat))
                #     .reshape(1, -1))
                # )
                test_prediction = self.svc.predict(test_features)
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(
                        draw_img, (xbox_left, ytop_draw + ystart),
                        (xbox_left + win_draw, ytop_draw + win_draw + ystart),
                        (0, 0, 255), 6)

        return draw_img


if __name__ == "__main__":
    for path in glob.glob('test_images/*.jpg'):
        svc_scaler = Path("data.p")
        if svc_scaler.is_file():
            result = pickle.load(open("data.p", 'rb'))
        else:
            result = svc.generate_svc()
            pickle.dump(result, open("data.p", 'wb'))
        # import ipdb; ipdb.set_trace()
        pipeline = Pipeline(*result)
        img = misc.imread(path)
        out_img = pipeline.find_cars(img)
        misc.imsave('test_images_output/' + Path(path).name, out_img)
