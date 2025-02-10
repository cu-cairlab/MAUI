**MAUI \- Modular Analytics of UAS Imagery**

This repository contains the codebase to implement the workflow described in MAUI: Modular Analytics of UAS Imagery for Specialty Crop Research.
Docker images for running the DeepLabv3 and SAM models, along with sample image datasets, can be accessed on [Box](https://cornell.app.box.com/folder/306827538774).

The modules are:

* Image Acquistision

* Image Preprocessing

* Plant Canopy Segmentation

* Trait Extraction and Export

**Image Acquisition**

MAUI is designed to support users in processing multispectral GEOTIFF images of specialty crops. Images can be acquired with the uncrewed aerial system (UAS) of choice. The images referenced by the code in this repository were acquired with the MicaSense RedEdge-MX Dual Camera System mounted on a DJI Matrice 600. Specific sensor information for the RedEdge-MX can be found [here](https://support.micasense.com/hc/en-us/articles/360049354874-Dual-Camera-System-FAQs). 
For a detailed description of the image acquisition protocol used to acquire the images in this repo, we refer readers to the MAUI manuscript. 

**Image Preprocessing**  
The image preprocessing module is customizable. The inputs are multispectral UAS images for multiple flights. Outputs are co-registered orthomosaics (one orthomosaic per flight). 

*Generate orthomosaics*  
As an example of how to generate orthomosaics, we include a guide to batch process UAS imagery in Agisoft Metashape. The guide is adapted from the [MicaSense RedEdge MX processing workflow](https://agisoft.freshdesk.com/support/solutions/articles/31000148780-micasense-rededge-mx-processing-workflow-including-reflectance-calibration-in-agisoft-metashape-pro). 

*Co-register orthomosaics*  
Once users obtain a set of orthomosaics, the images can optionally be co-registered using the [AROSICS](https://github.com/GFZ/arosics) tool. We refer users to the [AROSICS GitHub](https://github.com/GFZ/arosics) for complete documentation and instructions. 

**Plant Canopy Segmentation**   
Three methods for plant canopy segmentation are included in the corresponding sub-folders:

* Color space \- Otsu’s method  
* Supervised deep learning \- DeepLabv3  
* Vision foundation model \- SAM

All methods require multispectral GEOTIFF files as input. Two sample GEOTIFF files are included in the [data](https://cornell.app.box.com/folder/306827586573) subfolder on Box: vineyard_sample.tif and hemp_sample.tif, for a chardonnay vineyard and a hemp field, respectively. 

Documentation for each method is provided in the corresponding subfolders. Docker images and bash scripts are provided for the DeepLabv3 and SAM methods (Docker images can be downloaded from [Box](https://cornell.app.box.com/folder/306823112906)). For guidance on which method may be more appropriate for different kinds of specialty crop imagery, we refer users to the MAUI manuscript. 

**Trait Extraction and Export**  
Trait extraction is demonstrated for a sample vineyard image dataset. This folder contains a Jupyter notebook showing how to use the canopy mask generated in the segmentation module and a GEOJSON file (chard_panels_2m.geojson, included for user reference) to extract geo-referenced reflectance data for all crop pixels. Optionally, users can aggregate data by experimental unit (i.e. export average reflectance values per plant). The output of the trait extraction module is a CSV file with a row for each crop pixel and columns for experimental unit IDs, pixel coordinates, and reflectance values.

We also include the supplemental `post_to_panels_RTK.py` script, which can be used to generate vineyard panel geojsons from a set of RTK-GPS points. 

