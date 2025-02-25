import os 
import glob
import numpy as np
import rasterio as rio
import pandas as pd
import json
import pyproj
import matplotlib.pyplot as plt
from rasterio import plot
from rasterio.mask import raster_geometry_mask
from shapely.geometry import shape, MultiPolygon
from shapely.ops import transform
import geopandas as gpd
from geocube.api.core import make_geocube
import rioxarray as rx
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract georeferenced crop reflectance data")
    parser.add_argument('ortho', type=str, help="path to 10-band multispectral orthomosaic - .tif")
    parser.add_argument('mask', type=str, help="path to mask file - .png, .tif, .jpg")
    parser.add_argument('out_dir', type=str, help="output directory")
    parser.add_argument('metadata', type=str, help="path to metadata file with geometries of experimental units - .geojson or .shp")
    parser.add_argument('id_vars', nargs="+", help="list of id variables associated with each experimental unit e.g. 'plant_id','row_id', etc.")

    args = parser.parse_args()

ortho = args.ortho
mask = args.mask
out_dir = args.out_dir
metadata = args.metadata
id_vars = args.id_vars
###### Functions ##############
# Generate binary mask from multi-band .tif, .jpg, or .png file

def binary_mask_multi(mask_fp):
    '''This function turns a multi-band .tif, .jpg, or .png mask file into a single-band raster mask
    with unmasked pixels coded as 1s and masked pixels coded as 0s
    
    Inputs:
    mask_fp (str) : filepath to the mask .tif, .jpg, or .png file
    
    Returns:
    mask_arr_3d (np array) : 3-d numpy array of 0s and 1s'''

    if mask_fp[-3:]=='tif':
        with rio.open(mask_fp) as src:
            mask_arr = src.read()
            band_ct = mask_arr.shape[0]
    
            # get unique values for binary mask band (not necessarily 0 and 1)
            # binary mask band is the last band in the image
    
            mask_band = mask_arr[(band_ct-1)]
            mask_vals = np.unique(mask_band)
    
            # make binary mask 0s and 1s
            mask_arr_binary = mask_band*(1/(mask_vals[1]))
            mask_arr_binary[mask_arr_binary != 1] = 0
    
            mask_arr_3d = mask_arr_binary.reshape(1,mask_arr.shape[1],mask_arr.shape[2])
            return mask_arr_3d
            
    elif mask_fp[-3:]=='png':
        mask_arr = plt.imread(mask)
        band_ct = mask_arr.shape[2]
        mask_band = mask_arr[:,:,band_ct-1]
        mask_vals = np.unique(mask_band)
        mask_arr_binary = mask_band*(1/(mask_vals[-1]))
        mask_arr_binary[mask_arr_binary != 1] = 0
        mask_arr_3d = mask_arr_binary.reshape(1,mask_arr.shape[0],mask_arr.shape[1])
        return mask_arr_3d
        
    elif mask_fp[-3:]=='jpg':
        mask_arr = plt.imread(mask_fp)
        band_ct = mask_arr.shape[2]
        mask_band = mask_arr[:,:,-band_ct]
        mask_vals = np.unique(mask_band)
        mask_arr_binary = mask_band*(1/(mask_vals[-1]))
        mask_arr_binary[mask_arr_binary != 1] = 0
        mask_arr_3d = mask_arr_binary.reshape(1,mask_arr.shape[0],mask_arr.shape[1])
        return mask_arr_3d

    else:
        print ("mask file type is not supported. Supported file types are: .tif .png .jpg")

# Mask function

def mask_img(img_fp, mask_fp, output_dir):
    """
    This function masks a multispectral UAS image using a binary mask file
    The mask file must have the same dimensions and CRS as the UAV image.
    
    Inputs:
    img_fp (str) : filepath to the UAV image to be masked (.tif) 
    
    mask_fp (str) : filepath to the mask file (.tif, .png, .jpg)
    
    output_dir (str) : directory to store the masked .tif image (e.g. 'User/Desktop/')
    
    Returns:
    
    Filepath to the masked .tif file. 
    The masked .tif is saved to the output_dir and has the same dimensions and CRS as the original UAV image. 
    All masked pixels will have a value of 0 for all bands. Unmasked pixels will retain original band values.
    """ 
    
    mask_arr = binary_mask_multi(mask_fp) ## modify with appropriate helper fxn
    
    with rio.open(img_fp) as src:
        img_arr = src.read()
        masked_img_arr = mask_arr * img_arr
        
        kwargs = src.meta
        band_ct = masked_img_arr.shape[0]
        kwargs.update(dtype=rio.float32, count=band_ct)

        masked_img_fp = out_dir+'masked_'+ str(os.path.basename(img_fp))
        with rio.open(masked_img_fp,
                      'w', **kwargs) as dst:
            for b in range(masked_img_arr.shape[0]):
                dst.write_band(b+1, masked_img_arr[b].astype(rio.float32))
        
        return masked_img_fp
        #checks
        #print(masked_img_arr.shape)
        #plot.show(masked_img_arr[(band_ct-1)])

##########

# Mask the orthomosaic
mask_filepath = mask_img(ortho, mask, out_dir)

# Read the plant metadata file as a geodataframe
gdf = gpd.read_file(metadata)
# Reset index
gdf['index'] = gdf.index
# read in the masked image
img_data = rx.open_rasterio(mask_filepath)


vars = id_vars.append("index")

out_grid = make_geocube(
    vector_data=gdf,
    measurements=vars,
    like=img_data, # ensure the data are on the same grid
)

#This section is specific to the MicaSense Dual Camera System - modify according to your camera settings
cblue_444 = img_data[0]
blue_475 = img_data[1]
green_531 = img_data[2]
green_560 = img_data[3]
red_650 = img_data[4]
red_668 = img_data[5]
rededge_705 = img_data[6]
rededge_717 = img_data[7]
rededge_740 = img_data[8]
nir_842 = img_data[9]
    
    
band_dict = {'cblue_444':cblue_444, 'blue_475':blue_475, 'green_531':green_531,'green_560':green_560,
                'red_650':red_650,'red_668':red_668, 'rededge_705':rededge_705,'rededge_717':rededge_717,
                'rededge_740':rededge_740, 'nir_842':nir_842}

# merge the dfs together

for key, b in band_dict.items():
    out_grid[key] = (b.dims, b.values, b.attrs, b.encoding)

# Change 0 to NAN
out_grid_nans= out_grid.where(out_grid != 0)

# experimental unit ID
plant_id = list(id_vars)[0]

# Get a dataframe with per-pixel reflectance values
outgrid_df = out_grid_nans.to_dataframe()
outgrid_df.sort_values(by=id_vars, inplace=True)
outgrid_df.reset_index(inplace=True)
outgrid_df.dropna(subset=[plant_id,list(band_dict.keys())[0]], inplace=True) # remove pixels not associated with an ID or with missing reflectance data

# Export per-pixel data
outgrid_df.to_csv(out_dir+'/traits_per_pixel.csv',index=False)

# Calculate the average reflectance for each experimental unit
groupby_plantid = outgrid_df.groupby(outgrid_df.plant_id)
as_df = groupby_plantid.mean()
as_df.sort_values(by=[plant_id], inplace=True)
as_df.reset_index(inplace=True)
final_df = as_df.drop(['index'], axis=1)

# Export per-unit data
final_df.to_csv(out_dir+'/traits_per_unit.csv',index=False)