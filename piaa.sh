#!/bin/bash

MODEL=scclip
K=512

for DATANAME in voc12 voc07 coco nus; do
	echo "Running learn.py for $DATANAME..."
	python learn.py --dataname "$DATANAME" --model "$MODEL" --k "$K"

	echo "Running test.py for $DATANAME..."
	python test.py --dataname "$DATANAME" --model "$MODEL"
done

echo "All done!"


