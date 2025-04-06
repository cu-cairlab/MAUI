**Image Preprocessing**

The user inputs to the image pre-processing module are:
- geotiff files acquired during a UAS flight (.tif)
- ground control points (GCPs) for the study site (.csv)

For example, images acquired with the MicaSense Dual Camera System have the following file naming structure (one file for each waveband):

IMG_0000_1.tif

IMG_0000_2.tif

IMG_0000_3.tif

...

The GCP file contains the X,Y coordinates of the ground control points for the study site. In the MAUI manuscript, both study sites have 20 GCPs. A GCP file looks like this: 

### Sample Ground Control Points

| Point | Latitude (°) | Longitude (°) |
|-------|--------------|---------------|
| P1    | 40.6222      | -74.0603      |
| P2    | 40.7210      | -73.9487      |
| P3    | 40.5045      | -74.1078      |
| P4    | 40.8395      | -73.8324      |
| P5    | 40.6717      | -74.0114      |
| P6    | 40.4208      | -74.2069      |
| P7    | 40.9283      | -73.8849      |
| P8    | 40.4622      | -74.0482      |
| P9    | 40.6354      | -73.9516      |
| P10   | 40.7822      | -74.0944      |

