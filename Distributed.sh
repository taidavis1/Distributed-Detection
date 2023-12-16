#!/bin/bash

#SBATCH --job-name=taihenry
#SBATCH --nodes=${NUM_NODES}
#SBATCH --gpus-per-node=${NUM_GPUS}

module purge                        
ml cudatoolkit-standalone/11.8.0    
ml tensorflow


if [[-n "${NUM_NODES}" && "${NUM_NODES}" == 1]]; then
    !python /models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=/content/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8/pipeline.config \
    --model_dir=/content/training \
    --alsologtostderr

elif [[-n "${NUM_NODES}" && "${NUM_NODES}" > 1]]; then
    !python /models/research/object_detection/model_main_tf2.py --num_workers=${NUM_NODES} \
    --pipeline_config_path=/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8/pipeline.config \
    --model_dir=/content/training \
    --alsologtostderr
