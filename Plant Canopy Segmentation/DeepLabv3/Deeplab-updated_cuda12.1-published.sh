#!/bin/bash
#Deeplab COde bash
# Initialize variables if any
INPUT_DIR="" # Initialize the input directory
OUTPUT_DIR="" #Initialize the output_directory
TRAIN_MOD_DIR="" #Initialise the ignored directory 
EXP_NAME="" #Experiment_name for training
EPOCH_NUM="100" # Number of epochs for training
BATCH_SIZE="16" #Batch size for training 
BASE_RATE="0.000025" #Base learning rate 
CROP_SIZE="512"
RESUME="None"
WORKERS="3"
CLASS_NUM="2"
ALL= false
DATATYPE="grape"
PROG_VALUE="" # 1 for Convert_to_rgb , 2 for Tiling , 3 for training grape, 4 for implementation/inference
INF_MOD_PATH=""



while [[ "$#" -gt 0 ]]; do
    case $1 in
        -indir|--input-directory) INPUT_DIR="$2"; shift ;;
        -outdir|--output-directory) OUTPUT_DIR="$2"; shift ;;
		-modeldir|--model-directory) TRAIN_MOD_DIR="$2"; shift ;;
		-exp|--experiment_name) EXP_NAME="$2"; shift ;;
		-ep|--epochs) EPOCH_NUM="$2"; shift ;;
		-batch|--batch_size) BATCH_SIZE="$2" ; shift ;;
        -br|--base_rate) BASE_RATE="$2"; shift ;;
        -wk|--workers) WORKERS="$2"; shift ;;
        -prog|--prog-value) PROG_VALUE="$2"; shift ;; # 
		-inf|--inference_model) INF_MOD_PATH="$2" ; shift ;; #preferably within input folder
        -cr|--crop_size) CROP_SIZE="$2" ; shift ;;
        -class|--class_num) CLASS_NUM="$2" ; shift ;;
		-dt|--datatype) DATATYPE="$2" ; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate input for INPUT_DIR and PROG_VALUE. OUTPUT_DIR will be taken care of separately
if [[ -z "$INPUT_DIR" ]] || [[ -z "$PROG_VALUE" ]]; then
    echo "Usage: $0 -indir <input directory> -prog <Program_Value: 1 for RGBconversion, 2 for Tiling, 3 for Training, 4 for implementation> [-outdir <Output directory>]"
    exit 1
fi



if [[ -z "$OUTPUT_DIR" ]]; then
    CURRENT_DATE=$(date +%Y-%m-%d_%H-%M-%S)
    OUTPUT_DIR=$(pwd)/$CURRENT_DATE/train
fi

mkdir -p "$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR/rgb_png"

mkdir -p  "$OUTPUT_DIR/tiles"

mkdir -p  "$OUTPUT_DIR/inf/visual"

mkdir -p "$OUTPUT_DIR/inf/masks"

case $PROG_VALUE in 
    1)
    docker run \
            --gpus all\
            --rm\
			--workdir /home/ubuntu/canopyseg/ \
            --shm-size 8g\
            --env DISPLAY=$DISPLAY\
            --volume ${INPUT_DIR}:/home/ubuntu/input\
			--volume ${OUTPUT_DIR}:/home/ubuntu/canopyseg/train\
            --volume /tmp/.X11-unix:/tmp/.X11-unix\
            deeplab-segment:4.7_cuda12.1_published\
			/home/miniconda3/bin/python -u convert_to_rgb.py  /home/ubuntu/input train/rgb_png
			
    ;;
            

    2)
    docker run \
            --gpus all\
            --rm\
			--workdir /home/ubuntu/canopyseg/ \
            --shm-size 8g\
            --env DISPLAY=$DISPLAY\
            --volume ${INPUT_DIR}:/home/ubuntu/input\
			--volume ${OUTPUT_DIR}:/home/ubuntu/canopyseg/train\
            --volume /tmp/.X11-unix:/tmp/.X11-unix\
            deeplab-segment:4.7_cuda12.1_published\
			/home/miniconda3/bin/python -u ImageMask2Tiles_uniform.py  /home/ubuntu/input train/tiles
			
    ;;


    3) 
    docker run \
            --gpus all\
            --rm\
			--workdir /home/ubuntu/canopyseg/ \
            --shm-size 8g\
            --env DISPLAY=$DISPLAY\
            --volume ${INPUT_DIR}:/home/ubuntu/input\
			--volume ${OUTPUT_DIR}:/home/ubuntu/canopyseg/train\
			--volume ${TRAIN_MOD_DIR}:/home/ubuntu/canopyseg/ignored\
            --volume /tmp/.X11-unix:/tmp/.X11-unix\
            deeplab-segment:4.7_cuda12.1_published\
			/home/miniconda3/bin/python -u train_new.py --exp ${EXP_NAME} --epochs ${EPOCH_NUM} --batch_size ${BATCH_SIZE} --base_lr ${BASE_RATE} \
			--crop_size ${CROP_SIZE} --resume ${RESUME} --workers ${WORKERS} --num_classes ${CLASS_NUM} --datatype ${DATATYPE} \
			
			
	;;
			




    4)
    docker run \
            --gpus all\
            --rm\
			--workdir /home/ubuntu/canopyseg/ \
            --shm-size 8g\
            --env DISPLAY=$DISPLAY\
            --volume ${INPUT_DIR}:/home/ubuntu/input\
			--volume ${OUTPUT_DIR}:/home/ubuntu/canopyseg/train\
			--volume ${TRAIN_MOD_DIR}:/home/ubuntu/canopyseg/ignored\
            --volume /tmp/.X11-unix:/tmp/.X11-unix\
            deeplab-segment:4.7_cuda12.1_published\
			/home/miniconda3/bin/python -u inference_new.py train/rgb_png train/inf/masks train/inf/visual  train/${INF_MOD_PATH}
			
    ;;
	
	
	
	

    esac

    echo "Program Done"

    

