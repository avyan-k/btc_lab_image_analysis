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
for filename in $IMDIR*.ndpi; do
    echo "$filename"
    QuPath script "./qupath/scripts/Classify_Tumors.groovy" --image="$filename" > "./results/inference/$(basename "$filename"_inference_debug.txt)"
	python src/WSI_heatmap.py "$filename"
done