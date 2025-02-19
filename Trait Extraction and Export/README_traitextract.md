**Trait Extraction and Export**

Requirements:
- Python &ge;3.11

Packages:
- os 
- glob
- numpy
- rasterio
- pandas
- json
- pyproj
- matplotlib
- shapely
- geopandas
- geocube
- rioxarray
- argparse


*Jupyter notebook*

The jupyter notebook [traitextract_grapevine.ipynb](https://github.com/cu-cairlab/MAUI/blob/main/Trait%20Extraction%20and%20Export/traitextract_grapevine.ipynb)
demonstrates how to extract reflectance traits using grapevine as an example.
Users can download the notebook and use it directly by inputting filepaths to their own datasets. 
Sample data can also be downloaded from [Box](https://cornell.app.box.com/folder/308397071888). 

The files required to run the notebook are:
* 10-band multispectral orthomosaic (.tif)
* mask file (.tif, .png, or .jpg)
* metadata file (.geojson or .shp)


*Command line python script*

We also provide a python script ([traitextract.py](https://github.com/cu-cairlab/MAUI/blob/main/Trait%20Extraction%20and%20Export/traitextract.py)) 
to perform trait extraction from the command line.

The required arguments (in order) are:
- ```ortho``` (string): path to a 10-band orthomosaic .tif file
- ```mask``` (string): path to the mask file (.tif, .png, or .jpg) for the orthomosaic. ```ortho``` and ```mask``` dimensions must match.
- ```out_dir``` (string): path to the directory where output will be saved
- ```metadata``` (string): path to the georeferenced metadata file (.geojson or .shp). The metadata file must have at least one attribute field containing an identifier variable (e.g. unique ID, plant number, etc.)
- ```id_vars``` (string): one or more identifier variables contained in the metadata file (e.g. 'plant_id', 'row')

To use the script, follow these steps:

1. Download the ```traitextract.py``` file
2. Open Terminal (Mac) or other command line interface (Windows/Linux)
3. Navigate to the directory where the ```traitextract.py``` file is saved. Ex:
   ```cd User/Downloads/```
4. Type the following:
   ``` python traitextract.py path/to/orthomosaic path/to/mask path/to/output_directory path/to/metadata id_var1 id_var2 ...```

   Example usage:
   ``` python traitextract.py '/Downloads/chardonnay_20230705.tif' '/Downloads/chardonnay_20230705_mask.tif' '/Downloads/traitextract_outputs' '/Downloads/chard_panels.geojson' row panel```
   
