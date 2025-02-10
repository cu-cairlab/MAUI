import argparse
import numpy as np
from osgeo import gdal
import os
import cv2

os.environ['GDAL_DATA'] = '/home/miniconda3/share'
os.environ['PROJ_LIB'] = '/home/miniconda3/share/proj'
os.environ['PROJ_DATA'] = '/home/miniconda3/share/proj'

# Convert multispectral .tif to RGB .png
# band number is set for MicaSense RedEdgeMX Dual Camera System
# adjust band number as needed for images acquired with different sensors

def convert_to_rgb(root_dir, output_dir):
    for GeoTiff_fn in os.listdir(root_dir):
        GeoTiff_fp = os.path.join(root_dir, GeoTiff_fn)
        output_fp = os.path.join(output_dir, GeoTiff_fn[:-4] + '.png')
        raw = gdal.Open(GeoTiff_fp)
        width = raw.RasterXSize
        height = raw.RasterYSize
        bands = []
        for i in range(1, raw.RasterCount+1):
            band = raw.GetRasterBand(i).ReadAsArray(0, 0, width, height)
            bands.append(band)

        red_band = bands[4] #specific to RedEdgeMX Dual Camera
        green_band = bands[2] #specific to RedEdgeMX Dual Camera
        blue_band = bands[0] #specific to RedEdgeMX Dual Camera
        
        # example setting for image with different band numbering
        if GeoTiff_fn == 'chardonnay_20230719.tif':
            red_band = bands[2]
            green_band = bands[1]
            blue_band = bands[0]

        red_band = (((red_band - np.min(red_band)) / (np.max(red_band) - np.min(red_band))) * 255)
        green_band = (((green_band - np.min(green_band)) / (np.max(green_band) - np.min(green_band))) * 255)
        blue_band = (((blue_band - np.min(blue_band)) / (np.max(blue_band) - np.min(blue_band))) * 255)

        rgb_image = np.dstack((red_band, green_band, blue_band))
        rgb_image = rgb_image.astype(np.uint8)
        cv2.imwrite(output_fp, rgb_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert GeoTIFF images to RGB PNG images')
    parser.add_argument('root_dir', type=str, help='Root directory containing GeoTIFF images')
    parser.add_argument('output_dir', type=str, help='Output directory for RGB PNG images')
    args = parser.parse_args()

    convert_to_rgb(args.root_dir, args.output_dir)
