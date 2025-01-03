#!/bin/bash
if [[ $# != 1 ]]; then
	echo Error: Expected one input parameters.
	echo Usage: ./run_groovy_script.sh \<imagedirectory\> 
	exit 1
fi
CURRDIR=$(pwd)
IMDIR=$1
if [[ ! -d $IMDIR ]]; then
	echo Error: The directory to store \'$DIR\' does not exist. 
	exit 2
fi
if [[ ! -d "./qupath" ]]; then
	echo Error: The directory to QuPath project ./qupath does not exist. Please cd to repository directory to run groovy script. 
	exit 3
fi
source activate base
conda activate btc-labs
# Uncomment below if QUPATH throws error when initializeing pytorch engine
# conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9.7 -y
# cp ~/miniconda3/envs/btc-labs/lib/python3.10/site-packages/torch/lib/libcudnn.so.8 ~/.djl.ai/pytorch/2.0.1-cu118-linux-x86_64/libcudnn.so.8
for fullfilename in "$IMDIR"/*.ndpi; do
	echo "$fullfilename"
	filename=$(basename -- "$fullfilename")
	filename="${filename%.*}"
    QuPath script "./qupath/scripts/Classify_Tumors.groovy" --image="$fullfilename" > "./results/inference/$(basename "$filename"_inference_debug.txt)"
	python src/WSI_heatmap.py "$filename"
done