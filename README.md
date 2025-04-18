**MAUI \- Modular Analytics of UAS Imagery**

This repository contains the codebase to implement the workflow described in MAUI: Modular Analytics of UAS Imagery for Specialty Crop Research.
Docker images for running the DeepLabv3 and SAM models, along with sample image datasets, can be accessed on [Box](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309581343817).

![MAUI_forgithub](https://github.com/user-attachments/assets/d075510b-bac6-4a5f-9537-b29c2e19f8c1)


The modules are:

1. Image Acquistision

2. Image Preprocessing

3. Plant Canopy Segmentation

4. Trait Extraction and Export

**1. Image Acquisition**

MAUI is designed to support users in processing multispectral GEOTIFF images of specialty crops. Images can be acquired with the uncrewed aerial system (UAS) of choice. The images referenced by the code in this repository were acquired with the MicaSense RedEdge-MX Dual Camera System mounted on a DJI Matrice 600. Specific sensor information for the RedEdge-MX can be found [here](https://support.micasense.com/hc/en-us/articles/360049354874-Dual-Camera-System-FAQs). 
For a detailed description of the image acquisition protocol used to acquire the images in this repo, we refer readers to the MAUI manuscript. 

**2. Image Preprocessing**  
The image preprocessing module is customizable. The inputs are multispectral UAS images for multiple flights. Outputs are co-registered orthomosaics (one orthomosaic per flight). 

*Generate orthomosaics*  
As an example of how to generate orthomosaics, we include a guide to batch process UAS imagery in Agisoft Metashape. The guide is adapted from the [MicaSense RedEdge MX processing workflow](https://agisoft.freshdesk.com/support/solutions/articles/31000148780-micasense-rededge-mx-processing-workflow-including-reflectance-calibration-in-agisoft-metashape-pro). 

*Co-register orthomosaics*  
Once users obtain a set of orthomosaics, the images can optionally be co-registered using the [AROSICS](https://github.com/GFZ/arosics) tool. We refer users to the [AROSICS GitHub](https://github.com/GFZ/arosics) for complete documentation and instructions. 

**3. Plant Canopy Segmentation**   
Three methods for plant canopy segmentation are included in the corresponding sub-folders:

* Color space \- Otsu’s method  
* Supervised deep learning \- DeepLabv3  
* Vision foundation model \- SAM

All methods require GEOTIFF files as input. Sample GEOTIFF files are included in the [sample_data](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309581728621) subfolder on Box.

Documentation for each method is provided in the corresponding subfolders. Docker images and bash scripts are provided for the DeepLabv3 and SAM methods (Docker images can be downloaded from [Box](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309581343817)). For guidance on which method may be more appropriate for different kinds of specialty crop imagery, we refer users to the MAUI manuscript. 

**4. Trait Extraction and Export**  
Trait extraction is accomplished with the ```traitextract.py``` script. The functionality is demonstrated for a sample vineyard image dataset in the ```traitextract_grapevine.ipynb``` Jupyter notebook. The notebook shows how to use the canopy mask generated in the segmentation module and a GEOJSON metadata file  to extract geo-referenced reflectance data for all crop pixels. Users can also aggregate data by experimental unit (i.e. export average reflectance values per plant). The output of the trait extraction module is a CSV file with a row for each crop pixel and columns for experimental unit IDs, pixel coordinates, and reflectance values.

We also include the supplemental `post_to_panels_RTK.py` script, which can be used to generate vineyard panel geojsons from a set of RTK-GPS points. 

