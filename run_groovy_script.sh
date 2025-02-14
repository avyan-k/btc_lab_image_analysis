#!/bin/bash
if [[ $# != 2 ]]; then
	echo Error: Expected two input parameters.
	echo Usage: ./run_groovy_script.sh \<model name\> \<imagedirectory\> 
	exit 1
fi
CURRDIR=$(pwd)
MODEL=$1
if [[ ! -d "./results/training/torchscript_models/ResNet_Tumor/$MODEL" ]]; then
	echo Error: The model to store \'$MODEL\' does not exist. Please pick a valid model from:
	ls ./results/training/torchscript_models/ResNet_Tumor/
	exit 2
fi

IMDIR=$2
if [[ ! -d $IMDIR ]]; then
	echo Error: The directory to store \'$DIR\' does not exist. 
	exit 3
fi
if [[ ! -d "./qupath" ]]; then
	echo Error: The directory to QuPath project ./qupath does not exist. Please cd to repository directory to run groovy script. 
	exit 4
fi
for fullfilename in "$IMDIR"*.ndpi; do
	echo "$fullfilename"
	filename=$(basename -- "$fullfilename")
	filename="${filename%.*}"
    QuPath script "./qupath/scripts/Classify_Tumors.groovy" --image="$fullfilename" --args "$MODEL" > "./results/inference/$(basename "$filename"_"$MODEL"_inference_debug.txt)"
	python src/WSI_heatmap.py "$filename" "$MODEL"
done