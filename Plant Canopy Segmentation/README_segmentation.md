**Plant Canopy Segmentation**

Requirements:

* Compatible NVIDIA GPU, NVIDIA driver, NVIDIA CUDA Toolkit, and NDVIDIA cuDNN* 

* [Docker Desktop](https://docs.docker.com/desktop/)

*Users can check compatibility of all required NVIDIA tools [here](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html). 

<ins>Accessing Docker images</ins>

* The DeepLabv3 and SAM Docker images and sample datasets can be downloaded from Box [here](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309581343817). 

* Bash scripts to execute the Docker files are included in the subfolders in this repository and on Box ([MAUI_Modular_Analytics_of_UAS_Imagery > BashScripts](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309583722636)).

* Documentation describing each model and how to execute and configure the bash scripts is included in the corresponsing subfolders in this repository.

<ins>Usage example - hemp</ins>

1. Navigate to [MAUI_Modular_Analytics_of_UAS_Imagery > DockerImages](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309580730005)
   
2. Download segment-anyhemp_cuda12.1_published.tar
   
3. Navigate to [MAUI_Modular_Analytics_of_UAS_Imagery > BashScripts](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309583722636)
   
4. Download Segment-anything_updated_cuda12.1.sh
 
5. Read the instructions in [SAM_Segmentation_Doc.pdf](https://github.com/cu-cairlab/MAUI/blob/main/Plant%20Canopy%20Segmentation/SAM/SAM_Segmentation_Doc.pdf) (this repo). The documentation can also be accessed on Box at [MAUI_Modular_Analytics_of_UAS_Imagery > Documentation](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309581795265)

Note: The "Setup" section in the documentation provides detailed instructions for setup on different operating systems. For both Windows and Linux, users must install Docker Desktop. Installation instructions for different operating systems can be found [here](https://docs.docker.com/get-started/get-docker/). 

6. After loading the Docker image in Docker desktop (see sections 2 and 3 in [SAM_Segmentation_Doc](https://github.com/cu-cairlab/MAUI/blob/main/Plant%20Canopy%20Segmentation/SAM/SAM_Segmentation_Doc.pdf)), open your terminal application.

7. Navigate to the folder containing the downloaded bash script (Segment-anything_updated_cuda12.1.sh). Ex:
   ```cd Downloads/```

8. Follow the instructions in the documentation to run the canopy segmentation functions. The following commands show how to execute each function. References to specific files correspond to the [hemp sample dataset](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309582472093) available for download on Box.

ROI selection:

```bash Segment-anything_updated_cuda12.1.sh -indir Downloads/SAM -map 22_08_10 -prog 1```

DBScan:

```bash Segment-anything_updated_cuda12.1.sh -indir Downloads/SAM -map 22_08_10 -prog 2 -field_col  50 -f_id 3000 -row_col 15```

Co-registration:

```bash Segment-anything_updated_cuda12.1.sh -indir Downloads/SAM -map 22_08_10 -prog 4 -dates "22_08_03 22_08_18"```

Plant Extraction:

```bash Segment-anything_updated_cuda12.1.sh -indir Downloads/SAM -map 22_08_10 -prog 3 -dates "22_08_10 22_08_03 22_08_18" -sam on```

<ins>Usage example - grapevine</ins>

1. Navigate to [MAUI_Modular_Analytics_of_UAS_Imagery > DockerImages](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309580730005)
   
2. Download deeplab-segment_4.7_cuda12.1_published.tar
   
3. Navigate to [MAUI_Modular_Analytics_of_UAS_Imagery > BashScripts](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309583722636)
   
4. Download Deeplab-updated_cuda12.1-published.sh
 
5. Read the instructions in [DeepLab_Segmentation_Doc.pdf](https://github.com/cu-cairlab/MAUI/blob/main/Plant%20Canopy%20Segmentation/DeepLabv3/Deeplab_Segmentation_Doc.pdf) (this repo). The documentation can also be accessed on Box at [MAUI_Modular_Analytics_of_UAS_Imagery > Documentation](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309581795265)

Note: The "Setup" section in the documentation provides detailed instructions for setup on different operating systems. For both Windows and Linux, users must install Docker Desktop. Installation instructions for different operating systems can be found [here](https://docs.docker.com/get-started/get-docker/). 

6. After loading the Docker image in Docker desktop (see sections 2 and 3 in [DeepLab_Segmentation_Doc](https://github.com/cu-cairlab/MAUI/blob/main/Plant%20Canopy%20Segmentation/DeepLabv3/Deeplab_Segmentation_Doc.pdf)), open your terminal application.

7. Navigate to the folder containing the downloaded bash script (Deeplab-updated_cuda12.1-published.sh). Ex:
   ```cd Downloads/```

8. Follow the instructions in the documentation to run the canopy segmentation functions. The following commands show how to execute each function. References to specific files correspond to the [DeepLabv3/grapevine](https://cornell.app.box.com/s/vwb7pd4r546rfj0yrflyh4na5dj7ic97/folder/309584279948) sample dataset available for download on Box.

RGB conversion:

```bash Deeplab-updated_cuda12.1-published.sh -indir Downloads/grapevine -outdir Downloads/grapevine/output -prog 1```

Tiling:

```bash Deeplab-updated_cuda12.1-published.sh -indir Downloads/grapevine/output/rgb_png -outdir Downloads/grapevine/output -prog 2```

Training:

```bash Deeplab-updated_cuda12.1-published.sh -indir Downloads/grapevine -outdir Downloads/grapevine/output -prog 3 -exp experiment1_grapevine -model_dir Downloads/grapevine/output/model_dir -ep 30 -dt grape```

Inference:

```bash Deeplab-updated_cuda12.1-published.sh -indir Downloads/grapevine -outdir Downloads/grapevine/output -prog 4 -inf Downloads/grapevine/output/experiment1_grapevine/{XXX.pth}  -model_dir Downloads/grapevine/output/model_dir```


<ins>Additional scripts</ins>

The subfolders for DeepLabv3 and SAM in this repo contain selected scripts that are included in the Docker instances. These scripts are provided here for user reference and can be modified or used separately from the rest of the pipeline to execute individual functions. 


   
