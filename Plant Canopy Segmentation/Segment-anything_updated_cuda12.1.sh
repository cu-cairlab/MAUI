#!/bin/bash
#Segment Anything COde bash
# Initialize variables if any
INPUT_DIR="" # Initialize the input directory
MAP_NAME="" #Initialize the output_directory
OTHER_DATES=""
error=""
ALL= false
SAM_ON=""
PROG_VALUE=1 # 1 Roi Selection(must be precise as possible) , 2 for DBScan , 3 for plant extraction , 4 for co-registration, 5 for all
WOrK_DIR=""
WIDTH="160"
HEIGHT="160"
EXG_TH="0.2"
KERNEL_SIZE="5"
MIN_COUNT="0"
FIELD_NUM_COL="50"
F_ID="3000"




while [[ "$#" -gt 0 ]]; do
    case $1 in
        -indir|--input-directory) INPUT_DIR="$2"; shift ;; #working /parent directory. Should contain RGBmaps folder
        -map|--map-name) MAP_NAME="$2"; shift ;; #RGB map name. if the full name is 11-04-22_RGB.tif(recommended naming format), the map name would be 11-04-22
        -dates|--other-dates) OTHER_DATES="$2"; shift ;;
        -prog|--prog-value) PROG_VALUE="$2"; shift ;; # 1 for roi selection, 2 for dbscan, 3 plant extraction, 4 for co-registration
        -sam|--sam-segment) SAM_ON="$2"; shift ;; # turn sam-segmentation on or off. If any value is placed it turns on
        -w|--width) WIDTH="$2"; shift ;; #width of bounding box
        -h|--height) HEIGHT="$2" ; shift ;; #height of bounding box
        -exg_th|--exg_threshold) EXG_TH="$2"; shift ;; #threshold for ExGI map in dbscan
        -kernel|--kernel_size) KERNEL_SIZE="$2" ; shift ;; #dilation kernel size for dbscan
        -min|--min_counts) MIN_COUNT="$2" ; shift ;;
        -field_col|field_num_col) FIELD_NUM_COL="$2" ; shift ;; # number of columns to help assign correct id's in dbscan
        -f_id|first_id) F_ID="$2" ; shift ;; #assign correct plant_id in dbscan

        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate input for INPUT_DIR and PROG_VALUE. OUTPUT_DIR will be taken care of separately
if [[ -z "$INPUT_DIR" ]] || [[ -z "$PROG_VALUE" ]]; then
    echo "Usage: $0 -indir <input directory> -prog <Program_Value: 1 for ROI Selection, 2 for DBScan, 3 for Plant_extraction, 4 for implementation> [-outdir <Output directory>]"
    exit 1
fi

echo ${INPUT_DIR}

case ${PROG_VALUE} in 
    1)
    docker run \
            --gpus all\
            --rm\
            --shm-size 8g\
            --env DISPLAY=$DISPLAY\
            --volume ${INPUT_DIR}:/home/ubuntu/input\
            --volume /tmp/.X11-unix:/tmp/.X11-unix\
            segment-anyhemp:4.0_cuda12.1\
			/home/miniconda3/bin/python /home/ubuntu/Hemp_Characterization-master/roi_selection.py  --data_dir  /home/ubuntu/input/RGBmaps --map_name ${MAP_NAME}_RGB.tif
			
    ;;
            

    2)
    docker run \
            --gpus all\
            --rm\
            --shm-size 8g\
            --env DISPLAY=$DISPLAY\
            --volume ${INPUT_DIR}:/home/ubuntu/input\
            --volume /tmp/.X11-unix:/tmp/.X11-unix\
            segment-anyhemp:4.0_cuda12.1\
			/home/miniconda3/bin/python /home/ubuntu/Hemp_Characterization-master/dbscan_seg_plant.py  --data_dir  /home/ubuntu/input --data_date ${MAP_NAME} \
            --width ${WIDTH} --height ${HEIGHT} --exg_th ${EXG_TH} --kernel_size ${KERNEL_SIZE} --minimum_counts ${MIN_COUNT} --field_num_col ${FIELD_NUM_COL} --first_plant_id ${F_ID}
    ;;

    3)
    if [[ -z "$SAM_ON" ]]; then
    docker run \
            --gpus all\
            --rm\
            --shm-size 8g\
            --env DISPLAY=$DISPLAY\
            --volume ${INPUT_DIR}:/home/ubuntu/input\
            --volume /tmp/.X11-unix:/tmp/.X11-unix\
            segment-anyhemp:4.0_cuda12.1\
			/home/miniconda3/bin/python /home/ubuntu/Hemp_Characterization-master/extract_plants.py  --data_dir  /home/ubuntu/input --data_date ${MAP_NAME}  --data_date_list ${OTHER_DATES}
    else 
    docker run \
            --gpus all\
            --rm\
            --shm-size 8g\
            --env DISPLAY=$DISPLAY\
            --volume ${INPUT_DIR}:/home/ubuntu/input\
            --volume /tmp/.X11-unix:/tmp/.X11-unix\
            segment-anyhemp:4.0_cuda12.1\
			/home/miniconda3/bin/python /home/ubuntu/Hemp_Characterization-master/extract_plants.py  --data_dir  /home/ubuntu/input --data_date ${MAP_NAME}  --sam_segmentation --data_date_list ${OTHER_DATES}
    fi
    
    ;;
	
	
	4)
	
	
	
	readarray -d " " -t strarr <<< ${OTHER_DATES}
	
	for (( n=0; n < ${#strarr[*]}; n++))
	do
		 curr=${strarr[n]}
		 curr=$(echo $curr | tr -d '\n')
		 echo ${curr}
		 docker run \
            --gpus all\
            --rm\
            --shm-size 8g\
            --env DISPLAY=$DISPLAY\
            --volume ${INPUT_DIR}:/home/ubuntu/input\
            --volume /tmp/.X11-unix:/tmp/.X11-unix\
            segment-anyhemp:4.0_cuda12.1\
			/home/miniconda3/bin/arosics global /home/ubuntu/input/RGBmaps/${MAP_NAME}_RGB.tif /home/ubuntu/input/RGBmaps/${curr}_RGB.tif  -o /home/ubuntu/input/RGBmaps/Coregistered/"${strarr[n]}"_to_${MAP_NAME}_RGB.tif -fmt_out GTIFF -max_shift 200
			
	done
		

   
    ;;

    
	
	


   
    esac



    echo "Program Done."
