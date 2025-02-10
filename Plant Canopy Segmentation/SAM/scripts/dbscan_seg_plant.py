import os
os.environ['GDAL_DATA'] = '/home/miniconda3/share'
os.environ['PROJ_LIB'] = '/home/miniconda3/share/proj'
os.environ['PROJ_DATA'] = '/home/miniconda3/share/proj'
import cv2
import os.path as osp
import argparse
from osgeo import gdal, osr, ogr

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from sklearn.cluster import DBSCAN


def compute_ExG(selected_img, exg_th=0.2, kernel_size=5):
    # calculate ExG index image
    image_arr = selected_img.copy() / 255

    img_R = image_arr[:,:,0]
    img_G = image_arr[:,:,1]
    img_B = image_arr[:,:,2]

    img_r = img_R / (img_R + img_G + img_B)
    img_g = img_G / (img_R + img_G + img_B)
    img_b = img_B / (img_R + img_G + img_B)

    exg_img = 2*img_g - img_r - img_b # calcualte ExG index

    # calculate plant masks
    plant_mask = 255*(exg_img > exg_th).astype('uint8')
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)

    return exg_img, plant_mask


def dbscan_clustering(features, eps=8, min_samples=100):

    print('Start DBSCAN!! This may take a while')
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(features)
    print('Finished DBSCAN!')

    return dbscan


def get_args():
    parser = argparse.ArgumentParser('Grid-Generator')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--data_date', type=str, help='RGB map filename')
    parser.add_argument('--gene_map_for_date', type=str, help='generate Shapefile for specific dates')
    parser.add_argument('--width', type=int, help='width for the bounding box')
    parser.add_argument('--height', type=int, help='height for the bounding box')

    # ExGI parameters
    parser.add_argument('--exg_th', type=float, default=0.2, help='threshold for ExGI map')
    parser.add_argument('--kernel_size', type=int, default=5, help='dilation kernel size')
    # DBSCAN parameters
    parser.add_argument('--minimum_counts', type=int, default=0, help='minimum # pixels for valid clusters')
    # Plant ID assignment parameters
    parser.add_argument('--field_num_col', type=int, default=50, help='# rows/cols in the field to assign correct plant ID')
    parser.add_argument('--first_plant_id', type=int, default=3000, help='to assign correct plant ID')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    data_dir = args.data_dir
    img_date = args.data_date
    rgb_img_dir = osp.join(data_dir, 'RGBmaps')
    rgb_img_name = f'{img_date}_RGB.tif'
    rgb_img_file = osp.join(rgb_img_dir, rgb_img_name)

    selected_img_name = f'{img_date}_RGB_selected.png'
    selected_img_file = osp.join(rgb_img_dir, selected_img_name)

    selected_mask_name = f'{img_date}_RGB_mask.png'
    selected_mask_file = osp.join(rgb_img_dir, selected_mask_name)

    assert osp.exists(rgb_img_file), f'{rgb_img_file} Not Found'
    assert osp.exists(selected_mask_file), f'{selected_mask_file} Not Found'
    assert osp.exists(selected_img_file), f'{selected_img_file} Not Found'

    dbscan_weight_dir = osp.join(data_dir, 'DBSCAN_Weight')
    dbscan_weight_path = osp.join(dbscan_weight_dir, f"{img_date}_dbscan_weights.txt")
    os.makedirs(dbscan_weight_dir, exist_ok=True)

    shapefile_dir = osp.join(rgb_img_dir, 'Shapefile')
    os.makedirs(shapefile_dir, exist_ok=True)

    shapefile_name_date = args.gene_map_for_date if args.gene_map_for_date else rgb_img_name.split('.')[0]
    shapefile_name_poly = 'map_' + shapefile_name_date + '_poly.shp'
    shapefile_path_poly = osp.join(shapefile_dir, shapefile_name_poly)


    # thresholds for ExG compute
    exg_th = args.exg_th
    kernel_size = args.kernel_size
    # minimum pixel# for valid clusters
    minimum_counts = args.minimum_counts
    # bounding box size
    width = args.width if args.width else 160
    height = args.height if args.height else 160
    # TODO this depends on the field design
    # field col#
    field_num_col = args.field_num_col
    # plantid in metadata
    plant_id = args.first_plant_id

    ### =========================== ###
    ### ExG-based mask computation  ###
    ### =========================== ###
    selected_img = cv2.imread(selected_img_file, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = selected_img.shape
    exg_img, plant_mask = compute_ExG(selected_img, exg_th=exg_th, kernel_size=kernel_size)

    # manuscript figure
    # fig_filepath = osp.join(rgb_img_dir, f'{img_date}_plant_mask_grayscale.png')
    # plt.imshow(plant_mask, cmap='gray')
    # plt.axis('off')
    # plt.savefig(fig_filepath, bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.close()

    # plot original RGB and plant_mask
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
    axes[0].imshow(selected_img)
    axes[1].imshow(plant_mask)
    fig_filepath = osp.join(rgb_img_dir, f'{img_date}_plant_mask.png')
    fig.savefig(fig_filepath)
    plt.close(fig)

    ### =========================== ###
    ### DBSCAN clustering  ###
    ### =========================== ###
    row_indices, col_indices = np.where(plant_mask)
    features = np.column_stack((row_indices, col_indices))

    if os.path.exists(dbscan_weight_path):
        # Load DBSCAN weights (i.e., labels) from the file
        cluster_labels = np.loadtxt(dbscan_weight_path, dtype=int)
        print(f'Load DBSCAN labels from {dbscan_weight_path}')
    else:
        # DBSCAN for clustering
        dbscan = dbscan_clustering(features)
        np.savetxt(dbscan_weight_path, dbscan.labels_, fmt='%d')
        print(f"DBSCAN weights saved to {dbscan_weight_path}")
        # Get cluster labels
        cluster_labels = dbscan.labels_

    # Count the number of unique clusters (-1 denotes noise)
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
    valid_clusters = [l for l, c in zip(unique_labels[1:], label_counts[1:]) if c >= minimum_counts] # Exclude noise cluster (-1)
    viridis = mpl.colormaps['viridis'].resampled(len(valid_clusters))
    # Generate random colors for each cluster
    cluster_colors = [viridis(label)[:-1] for label in unique_labels]

    dbscan_clustered_img = selected_img.copy().astype(float)
    for i, label in enumerate(valid_clusters):
        # Get the indices of pixels belonging to the current cluster
        cluster_indices = np.where(cluster_labels == label)[0]
        # Convert the flattened indices back to the original image indices
        row_indices, col_indices = features[cluster_indices, 0], features[cluster_indices, 1]
        # Assign random color to these pixels in the clustered image
        dbscan_clustered_img[row_indices, col_indices] = viridis(label+1)[:-1]

    # manuscript figure
    # fig_filepath = osp.join(rgb_img_dir, f'{img_date}_dbscan_clusters.png')
    # plt.imshow(dbscan_clustered_img)
    # plt.axis('off')
    # plt.savefig(fig_filepath, bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.close()

    # plot
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(40,20))
    axes[0].imshow(selected_img)
    axes[1].imshow(dbscan_clustered_img)
    axes[2].imshow(selected_img)

    # manuscript figure
    fig2, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    ax.imshow(selected_img)
    
    counter = 0
    bbox_dict = {}
    for i, label in enumerate(valid_clusters):
        # Get the indices of pixels belonging to the current cluster
        cluster_indices = np.where(cluster_labels == label)[0]
        # Convert the flattened indices back to the original image indices
        row_indices, col_indices = features[cluster_indices, 0], features[cluster_indices, 1]
        # Add text annotation for the cluster
        centroid_x = int(np.mean(col_indices))
        centroid_y = int(np.mean(row_indices))
        axes[1].text(centroid_x, centroid_y, label, color='red', fontsize=12, ha='center', va='center')

        min_row, min_col = centroid_y - height//2, centroid_x - width//2
        max_row, max_col = centroid_y + height//2, centroid_x + width//2
        center = (centroid_x, centroid_y)

        counter = counter + 1
        ## X, Y, CX, CY, W, H
        bbox_dict[label] = (min_col, min_row, min_col+width//2, min_row+height//2, width, height)

        # Plot bounding box on the image
        rect = Rectangle((min_col, min_row), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        # axes[2].add_patch(rect)
        axes[2].text(min_col, min_row, label, color='blue', fontsize=12, ha='center', va='center')

    axes[1].set_title(f'#clusters: {counter}', fontsize=20)
    for i in range(1,field_num_col+1):
        axes[2].axvline(x=(img_width//field_num_col)*i, color='black', linestyle='--', linewidth=2)
    fig_filepath = osp.join(rgb_img_dir, f'{img_date}_dbscan_clustering.png')
    fig.savefig(fig_filepath)
    plt.close(fig)

    ax.set_axis_off()
    fig_filepath = osp.join(rgb_img_dir, f'{img_date}_bounding_box.png')
    fig2.savefig(fig_filepath, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig2)

    ### =========================== ###
    ### Reorder BBox to match metadata  ###
    ### =========================== ###
    """
    This reordering depends on how the plants were ordered in the field.
    The mapping generated by the code cannot guarantte 100% accuracy because of plants being missed by DBSCAN clustering.
    Make sure to do a manual sanity-check!!
    """

    col_interval = img_width // field_num_col

    # X, Y, CX, CY, W, H
    min_col_row = np.asarray(list(bbox_dict.values()))
    # Sort based on the X coordinate (col)
    sorted_indices = np.argsort(min_col_row[:, 0])
    sorted_x_bbox = min_col_row[sorted_indices]

    sorted_bbox = []
    for i in range(field_num_col):
        # Select all plants whose center is between X=0 and X=0+col_interval
        x_left_bound = col_interval * i
        x_right_bound = col_interval * (i+1)
        selected_bbox = sorted_x_bbox[(sorted_x_bbox[:, 2]>x_left_bound)&(sorted_x_bbox[:, 2]<=x_right_bound), :]
        # Sort based on the Y coordinate (row)
        sorted_indices = np.argsort(selected_bbox[:, 1])[::-1]
        sorted_bbox.append(selected_bbox[sorted_indices])

    sorted_bbox_concat = np.concatenate(sorted_bbox, axis=0)

    # plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(40,20))
    axes[0].imshow(selected_img)
    axes[1].imshow(selected_img)
    for i, bbox in enumerate(sorted_bbox_concat):
        min_col, min_row, center_col, center_row, width, height = bbox
        # Plot bounding box on the image
        rect = Rectangle((min_col, min_row), width, height, fill=False, edgecolor='red', linewidth=2)
        axes[1].add_patch(rect)
        axes[1].text(min_col, min_row, i+1, color='blue', fontsize=12, ha='center', va='center')
    fig_filepath = osp.join(rgb_img_dir, f'{img_date}_ordered_individual_plant_map.png')
    fig.savefig(fig_filepath)


    ### =========================== ###
    ### Convert pixel-coor to utm  ###
    ### =========================== ###
    """
    All previous operations were done on the manually selected RGB maps which is only 
    a portion of the original ortho map. The following code makes sure
    1. BBoxes could be correctly applied to the original ortho map.
    2. Conversion from pixel-coor to utm
    """

    # read original RGB image
    dataset = gdal.Open(rgb_img_file, gdal.GA_ReadOnly)
    # print(type(dataset))

    band_b = dataset.GetRasterBand(1)
    band_g = dataset.GetRasterBand(2)
    band_r = dataset.GetRasterBand(3)
    arr_b = band_b.ReadAsArray()
    arr_g = band_g.ReadAsArray()
    arr_r = band_r.ReadAsArray()

    # projection dictionary
    map_origin = np.array([dataset.GetGeoTransform()[0],
                            dataset.GetGeoTransform()[3]])
    east_resolution = dataset.GetGeoTransform()[1]
    north_resolution = dataset.GetGeoTransform()[5]
    projection_dict = {
        'east_res': east_resolution,
        'north_res': north_resolution,
        'origin': map_origin
    }

    # assemble map image
    rgb_image = np.stack([arr_r, arr_g, arr_b], axis=2)

    mask = cv2.imread(selected_mask_file, cv2.COLOR_BGR2RGB)
    # print(mask.shape)

    row_indices, col_indices = np.where(mask[...,0])
    top_left_x, top_left_y = np.min(col_indices), np.min(row_indices)
    # print(f'top left coor (X, Y): ({top_left_x, top_left_y})')

    # 
    sorted_bbox_concat_ori = sorted_bbox_concat.copy()
    sorted_bbox_concat_ori[:, 0] = sorted_bbox_concat_ori[:, 0] + top_left_x
    sorted_bbox_concat_ori[:, 1] = sorted_bbox_concat_ori[:, 1] + top_left_y
    sorted_bbox_concat_ori[:, 2] = sorted_bbox_concat_ori[:, 2] + top_left_x
    sorted_bbox_concat_ori[:, 3] = sorted_bbox_concat_ori[:, 3] + top_left_y

    # plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(40,20))
    axes[0].imshow(rgb_image)
    axes[1].imshow(rgb_image)
    for i, bbox in enumerate(sorted_bbox_concat_ori):
        min_col, min_row, center_col, center_row, width, height = bbox
        # Plot bounding box on the image
        rect = Rectangle((min_col, min_row), width, height, fill=False, edgecolor='red', linewidth=2)
        axes[1].add_patch(rect)
        axes[1].text(min_col, min_row, i+1, color='blue', fontsize=12, ha='center', va='center')
    fig_filepath = osp.join(rgb_img_dir, f'{img_date}_bbox_on_ortho.png')
    fig.savefig(fig_filepath)

    # map origin and pixel resolution
    map_origin = projection_dict['origin']
    east_resolution = projection_dict['east_res']
    north_resolution = projection_dict['north_res']
    # pixel to utm transformation matrix
    px_utm_matrix = np.array([east_resolution, 0, 0, north_resolution]).reshape(2, 2)

    # calculate column (north-south) heading
    col_headings = []
    for col_id in range(field_num_col):
        # X, Y, CX, CY, W, H
        col_pts = sorted_bbox[col_id][:, 2:4] + (top_left_x, top_left_y)
        col_pts_utm = np.dot(col_pts, px_utm_matrix) + map_origin # utm coordinates
        col_start_pt = col_pts_utm[-1, :] # southmost point
        col_end_pt = col_pts_utm[0, :] # northmost point
        col_heading = math.atan(
            (col_end_pt[1] - col_start_pt[1]) / (col_end_pt[0] - col_start_pt[0])) / math.pi*180
        col_headings.append(col_heading)

    avg_col_heading = np.array(col_headings).mean()
    bbox_rotation_angle = avg_col_heading - 90
    # (x', y') = (x, y) dot (bbox_rotation_matrix)
    bbox_rotation_matrix = np.array(
        [math.cos(math.radians(bbox_rotation_angle)), 
        math.sin(math.radians(bbox_rotation_angle)), 
        -math.sin(math.radians(bbox_rotation_angle)), 
        math.cos(math.radians(bbox_rotation_angle))]).reshape(2,2)

    # convert from pixel-coor to field-coor
    bbox_width = width//100
    bbox_height = height//100

    plant_bbox = []
    for bbox in sorted_bbox_concat:
        plant_center = np.dot(bbox[2:4] + (top_left_x, top_left_y), px_utm_matrix) + map_origin
        # calculate vertices of the plant bounding box given the plant center
        # bottom-left, bottom-right, ..., -- counter-clockwise
        v1 = np.dot(np.array([-bbox_width/2, bbox_height/2]), bbox_rotation_matrix) + plant_center
        v2 = np.dot(np.array([bbox_width/2, bbox_height/2]), bbox_rotation_matrix) + plant_center
        v3 = np.dot(np.array([bbox_width/2, -bbox_height/2]), bbox_rotation_matrix) + plant_center
        v4 = np.dot(np.array([-bbox_width/2, -bbox_height/2]), bbox_rotation_matrix) + plant_center

        plant_bbox.append([v1, v2, v3, v4, plant_center])

    plant_bbox_res = {'column_headings': col_headings, 
                    'avg_col_heading': avg_col_heading,
                    'plant_bbox': plant_bbox
                    }

    # spatial reference information
    srs_txt = osr.SpatialReference(
        wkt=dataset.GetProjection()).GetAttrValue('AUTHORITY', 1)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(srs_txt))

    # generate polygon shapefile
    try:
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.CreateDataSource(shapefile_path_poly)
        layer = ds.CreateLayer('', srs, geom_type=ogr.wkbMultiPolygon)
        # create attributes
        layer.CreateField(ogr.FieldDefn('plant_id', ogr.OFTInteger)) # plant id
        # layer.CreateField(ogr.FieldDefn('col_id', ogr.OFTInteger)) # column id
        # layer.CreateField(ogr.FieldDefn('row_id', ogr.OFTInteger)) # row id
        layer.CreateField(ogr.FieldDefn('genotype', ogr.OFTString)) # genotype
        layer.CreateField(ogr.FieldDefn('c_east', ogr.OFTReal)) # plant center utm east
        layer.CreateField(ogr.FieldDefn('c_north', ogr.OFTReal)) # plant center utm north
        
        defn = layer.GetLayerDefn()

        # create plant polygons (bounding boxes)
        for idx, bbox_pts in enumerate(plant_bbox):
            bbox_pts = np.concatenate(bbox_pts)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            # add four vertices
            # v1, v2, v3, v4, plant_center
            for pt_id in range(4):
                ring.AddPoint(bbox_pts[pt_id*2], bbox_pts[pt_id*2+1])
            # close the polygon
            ring.AddPoint(bbox_pts[0], bbox_pts[1])
            # generate polygon shape
            plant_poly = ogr.Geometry(ogr.wkbPolygon)
            plant_poly.AddGeometry(ring)
            plant_poly_txt = plant_poly.ExportToWkt()
            # print(plant_poly_txt)
            # Create a new feature (attribute and geometry)
            feat = ogr.Feature(defn)
            feat.SetField('plant_id', plant_id)
            # feat.SetField('col_id', col_id+1)
            # feat.SetField('row_id', num_row-row_id)
            feat.SetField('c_east', bbox_pts[8])
            feat.SetField('c_north', bbox_pts[9])
            plant_id = plant_id + 1
            # Make a geometry, from Shapely object
            geom = ogr.CreateGeometryFromWkt(plant_poly_txt)
            feat.SetGeometry(geom)
            layer.CreateFeature(feat)
            ring = feat = geom = None
        print(f'Save Shapefile to {shapefile_path_poly}')
    except:
        print('Error raised in generating Shapefile!!')
    finally:
        ds = layer = feat = geom = None
