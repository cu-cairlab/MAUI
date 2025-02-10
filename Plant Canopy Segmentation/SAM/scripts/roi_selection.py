import cv2
import os
import os.path as osp
import argparse

import numpy as np

from utils import __select_roi, wnd_show

# Remember to change to your Anaconda3 folder
os.environ['PROJ_LIB'] = '/home/miniconda3/share/proj'
os.environ['GDAL_DATA'] = '/home/miniconda3/share'
os.environ['OMP_NUM_THREADS'] = '2'

from osgeo import gdal


def get_args():
    parser = argparse.ArgumentParser('Grid-Generator')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--map_name', type=str, help='RGB map filename')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    map_dir = args.data_dir
    map_name = args.map_name
    map_file = osp.join(map_dir, map_name)
    assert osp.exists(map_file), f"{map_file} Not Found"
    map_mask_name = map_name.split('.')[0] + '_mask.png'
    map_mask_file = osp.join(map_dir, map_mask_name)

    # read map image
    dataset = gdal.Open(map_file, gdal.GA_ReadOnly)
    print(type(dataset))
    # Note GetRasterBand() takes band no. starting from 1 not 0
    band_b = dataset.GetRasterBand(1)
    band_g = dataset.GetRasterBand(2)
    band_r = dataset.GetRasterBand(3)
    arr_b = band_b.ReadAsArray()
    arr_g = band_g.ReadAsArray()
    arr_r = band_r.ReadAsArray()
    # assemble map image
    map_image = np.stack([arr_r, arr_g, arr_b], axis=2)
    # select or load mask for the map image
    if osp.exists(map_mask_file):
        print(f'{map_mask_file} Found!')
    else:
        src_img = map_image
        save_dir = map_dir
        input_name = map_name
        resize_factor = 1/4

        resized_img = src_img.copy()
        resized_img = cv2.resize(resized_img, (int(src_img.shape[1]*resize_factor), int(src_img.shape[0]*resize_factor)))
        selected_pts = []
        selected_mask = np.zeros(src_img.shape, dtype=np.int8)
        params = [src_img, resized_img, selected_pts, selected_mask]

        cv2.imshow(wnd_show, resized_img)
        cv2.setMouseCallback(wnd_show, __select_roi, param=params)
        print("[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: determine the ROI area, esc: to exit window. Press after pressing middle button")

        while cv2.getWindowProperty(wnd_show, 0) >= 0:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        # export mask for future uses
        non_zero_indices = np.nonzero(params[3])
        non_zero_y_indices = non_zero_indices[0]
        non_zero_x_indices = non_zero_indices[1]
        y1 = np.min(non_zero_y_indices); y2 = np.max(non_zero_y_indices)
        x1 = np.min(non_zero_x_indices); x2 = np.max(non_zero_x_indices)
        selected_img = src_img[y1:y2, x1:x2]

        new_mask = np.zeros(src_img.shape, dtype=np.uint8)
        new_mask[y1:y2, x1:x2] = 255
        image_name = input_name.split('.')[0] + '_selected.png'
        image_path = osp.join(save_dir, image_name)
        cv2.imwrite(image_path, selected_img)

        mask_name = input_name.split('.')[0] + '_mask.png'
        mask_path = osp.join(save_dir, mask_name)
        cv2.imwrite(mask_path, new_mask)
