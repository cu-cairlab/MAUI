import os
import os.path as osp
os.environ['PROJ_LIB'] = '/home/miniconda3/share/proj'
os.environ['PROJ_DATA'] = '/home/miniconda3/share/proj'
import argparse

import fiona
import rasterio
import rasterio.mask

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from osgeo import gdal

from segment_anything import sam_model_registry, SamPredictor
from dbscan_seg_plant import compute_ExG


def find_rectangle_shape(shape):

    # v1-bottom-left, v2-bottom-right, ..., -- counter-clockwise
    v1 = shape["coordinates"][0][0]
    v2 = shape["coordinates"][0][1]
    v3 = shape["coordinates"][0][2]
    v4 = shape["coordinates"][0][3]

    # Extract x and y coordinates from vertices
    x_values = [v[0] for v in [v1, v2, v3, v4]]
    y_values = [v[1] for v in [v1, v2, v3, v4]]

    # Determine the minimum and maximum x and y coordinates
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    # Create a new dictionary for the rectangular shape
    rectangular_shape = {
        "type": shape["type"],
        "coordinates": [
            [
                (min_x, max_y),
                (max_x, max_y),
                (max_x, min_y),
                (min_x, min_y),
                (min_x, max_y),
            ]
        ],
    }

    return rectangular_shape


def load_sam(sam_checkpoint):
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")

    predictor = SamPredictor(sam)

    return predictor


def write_geotiff(filename, arr, in_ds):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    # if out_ds is not None:
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)


def get_args():
    parser = argparse.ArgumentParser("Grid-Generator")
    parser.add_argument("--data_dir", type=str, help="data directory")
    parser.add_argument("--data_date", type=str, help="RGB map filename")
    parser.add_argument("--data_date_list", nargs="+", help="list of data date")
    parser.add_argument(
        "--sam_segmentation",
        action="store_true",
        help="use SAM for semantic segmentation",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    data_dir = args.data_dir
    rgb_map_dir = osp.join(data_dir, "RGBmaps")
    coregistered_dir = osp.join(rgb_map_dir, "Coregistered")
    shapefile_dir = osp.join(rgb_map_dir, "Shapefile")
    ms_map_dir = osp.join(data_dir, "MSmaps")
    data_date_list = (
        args.data_date_list
    )  # ["22_07_26", "22_08_03", "22_08_10", "22_08_18"]
    reference_date = args.data_date

    sam_checkpoint = "/home/ubuntu/Hemp_Characterization-master/sam_vit_h_4b8939.pth" #path of the SAM training model you want to use
    assert osp.exists(sam_checkpoint), f"{sam_checkpoint} Not Found"

    shapefile_name = f"map_{reference_date}_RGB_poly.shp"
    shapefile_path = osp.join(shapefile_dir, shapefile_name)
    print(f"Use Shapefile {shapefile_path}")

    predictor = None
    if args.sam_segmentation:
        predictor = load_sam(sam_checkpoint)
        print(f"Load {sam_checkpoint} for zero-shot segmentation")

    for data_date in data_date_list:

        if data_date == reference_date:
            rgb_img_path = osp.join(rgb_map_dir, f"{data_date}_RGB.tif")
        else:
            rgb_img_path = osp.join(
                coregistered_dir, f"{data_date}_to_{reference_date}_RGB.tif"
            )
            print(f"Use co-registered ortho RGB map {rgb_img_path}")

        assert osp.exists(rgb_img_path), f"{rgb_img_path} Not Found"

        ms_map_name = f"{data_date}_MS.tif"
        ms_map_path = osp.join(ms_map_dir, ms_map_name)
        print(f"Processing {data_date}")

        # create save dirs
        plant_img_dir = osp.join(data_dir, "PlantImages")
        plant_png_dir = osp.join(plant_img_dir, "RGB", data_date)
        plant_tif_dir = osp.join(plant_img_dir, "TIF", data_date)
        plant_mask_dir = osp.join(plant_img_dir, "Mask", data_date)
        plant_mask_tif_dir = osp.join(plant_img_dir, "Mask_TIF", data_date)
        plant_debug_dir = osp.join(plant_img_dir, "Debug", data_date)
        os.makedirs(plant_png_dir, exist_ok=True)
        os.makedirs(plant_tif_dir, exist_ok=True)
        os.makedirs(plant_mask_dir, exist_ok=True)
        os.makedirs(plant_mask_tif_dir, exist_ok=True)
        os.makedirs(plant_debug_dir, exist_ok=True)

        with fiona.open(shapefile_path, "r") as shapefile: #open shapefile generated from dbscan script
            shapes = [feature["geometry"] for feature in shapefile]
            properties = [feature["properties"] for feature in shapefile]

        # crop BBoxes from original ortho and save as TIF
        with rasterio.open(rgb_img_path) as src:
            for shape, property_ in zip(shapes, properties):

                # Create a new dictionary for the rectangular shape
                rectangular_shape = find_rectangle_shape(shape)

                out_image, out_transform = rasterio.mask.mask(
                    src, [rectangular_shape], crop=True
                )

                out_meta = src.meta
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    }
                )

                plant_id = property_["plant_id"]

                tif_filepath = osp.join(plant_tif_dir, f"plant{plant_id}.tif")
                if os.path.exists(tif_filepath):
                    print(f"{tif_filepath} Already Exists - Skip")
                    continue

                with rasterio.open(tif_filepath, "w", **out_meta) as dest:
                    dest.write(out_image)

        # Iterate over all TIF files and save as RGB images
        for index, filename in enumerate(os.listdir(plant_tif_dir)):
            filepath = osp.join(plant_tif_dir, filename)
            plant_id = filename[5:-4]
            # Check if the current path is a file (not a subdirectory)
            assert osp.isfile(filepath) and filepath.endswith(
                ".tif"
            ), f"Invalid {filepath}!"

            raw = gdal.Open(filepath)
            width = raw.RasterXSize
            height = raw.RasterYSize
            imgarr = raw.ReadAsArray()
            # Read the bands as an array
            bands = []
            for i in range(1, 4):  # Assuming bands are numbered from 1 to 10
                band = raw.GetRasterBand(i).ReadAsArray(0, 0, width, height)
                bands.append(band)

            # TODO Each RGB is supposed to compose of two bands?
            # Select three bands for RGB representation (adjust the indices as needed)
            # red_band = bands[5]
            # green_band = bands[3]
            # blue_band = bands[1]

            red_band = bands[2]
            green_band = bands[1]
            blue_band = bands[0]

            # Normalize the pixel values to the range [0, 255]
            red_band = (
                (red_band - np.min(red_band)) / (np.max(red_band) - np.min(red_band))
            ) * 255
            green_band = (
                (green_band - np.min(green_band))
                / (np.max(green_band) - np.min(green_band))
            ) * 255
            blue_band = (
                (blue_band - np.min(blue_band))
                / (np.max(blue_band) - np.min(blue_band))
            ) * 255

            # Create an RGB image by stacking the bands and converting to uint8
            rgb_arr = np.dstack((red_band, green_band, blue_band)).astype(np.uint8)
            rgb_img = Image.fromarray(rgb_arr)
            rgb_filepath = osp.join(plant_png_dir, filename.replace(".tif", ".png"))
            if os.path.exists(rgb_filepath):
                print(f"{rgb_filepath} Already Exists - Skip")
            else:
                rgb_img.save(rgb_filepath)

            # SAM segmentation
            if predictor is not None:
                exg_img, exg_mask = compute_ExG(rgb_arr)
                row_indices, col_indices = np.where(exg_mask)
                valid_coor = np.column_stack((row_indices, col_indices))  # (Y,X)
                mean_coor = np.mean(valid_coor, axis=0)

                masks = predictor.set_image(rgb_arr)

                # (X, Y)
                input_point = np.array([mean_coor[::-1]])
                input_label = np.array([1])

                sam_masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                best_sam_mask = sam_masks[np.argmax(scores)]

                # Combine ExG mask and SAM mask to avoid shade
                final_mask = np.logical_and(best_sam_mask, exg_mask).astype(np.uint8)

                # Save 3 masks - ExG, SAM, and combined
                plant_mask_path = osp.join(plant_mask_dir, f"Plant{plant_id}.npy")
                np.save(plant_mask_path, final_mask)
                plant_mask_path = osp.join(plant_mask_dir, f"Plant{plant_id}_ExG.npy")
                np.save(plant_mask_path, exg_mask)
                plant_mask_path = osp.join(plant_mask_dir, f"Plant{plant_id}_SAM.npy")
                np.save(plant_mask_path, best_sam_mask)

                if index % 10 == 0:
                    plant_plt_path = osp.join(
                        plant_debug_dir, f"Plant{plant_id}_Mask.png"
                    )
                    # plot
                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
                    axes[0][0].imshow(rgb_arr)
                    axes[0][0].set_title(f"RGB Image")

                    axes[0][1].imshow(best_sam_mask)
                    axes[0][1].set_title(f"SAM Score: {np.max(scores):.3f}")

                    axes[1][0].imshow(exg_mask)
                    axes[1][0].set_title(f"ExG Mask")

                    axes[1][1].imshow(final_mask)
                    axes[1][1].set_title(f"Final Mask")
                    plt.savefig(plant_plt_path, dpi=300)
                    plt.close()

                plant_mask_tif_path = osp.join(
                    plant_mask_tif_dir, f"Plant{plant_id}_Mask.tif"
                )
                write_geotiff(plant_mask_tif_path, final_mask, raw)

                plant_mask_tif_path = osp.join(
                    plant_mask_tif_dir, f"Plant{plant_id}_ExG_Mask.tif"
                )
                write_geotiff(plant_mask_tif_path, exg_mask, raw)

                plant_mask_tif_path = osp.join(
                    plant_mask_tif_dir, f"Plant{plant_id}_SAM_Mask.tif"
                )
                write_geotiff(plant_mask_tif_path, best_sam_mask, raw)
