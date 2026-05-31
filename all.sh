#!/bin/bash


echo "Experimental Results" > result.txt
echo "====================" >> result.txt


for DATANAME in voc12 voc07 coco nus; do
    for MODEL in scclip itaclip sclip; do
        echo "----------------------------------------"
        echo "🚀 Running DATANAME=${DATANAME} | MODEL=${MODEL}"
        
        python learn.py --dataname "$DATANAME" --model "$MODEL" --k 512
        
        if [ $? -ne 0 ]; then
            echo "❌ ERROR: learn.py failed for ${DATANAME} ${MODEL}"
            echo "${DATANAME} - ${MODEL}: ERROR in learn.py" >> result.txt
            continue
        fi


        echo "Testing..."
        OUTPUT=$(python test.py --dataname "$DATANAME" --model "$MODEL" | grep -E "^mAP [0-9]+\.[0-9]+")
        
        if [ -z "$OUTPUT" ]; then
            echo "❌ ERROR: test.py failed or mAP not found for ${DATANAME} ${MODEL}"
            echo "${DATANAME} - ${MODEL}: ERROR or mAP not found" >> result.txt
        else
            echo "✅ Result: ${OUTPUT}"
            echo "${DATANAME} - ${MODEL} : ${OUTPUT}" >> result.txt
        fi

    done
done

echo "🎉 All tests completed! Please check result.txt."
