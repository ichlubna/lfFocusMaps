#!/bin/bash
sleep 7h
FILE="./noReferenceResults.txt"

runTest() {
    SCENE=$1
    RANGE=$2
    ASPECT=$3
    METHOD=$4
    rm -rf ./data/*
    mkdir ./data/data_down
    python ./renameLFImages.py ./dataset/nonanim/${SCENE}/001/ ./data/ 0 0 15 15 0 1 27
    if [[ $METHOD == "PYR" ]]; then
        python ./preprocessing/launchPreprocess.py ./data/data ./data/data_down GAUSSIAN_LIGHT_HALF
    fi

    #FAST=""
    #if [[ $METHOD == "BFF" ]]; then
    #    METHOD="BF"
        FAST="-f"
    #fi

    for V in 0.143_0.643 0.357_0.143 0.5_0.5 0.643_0.857 0.857_0.357; do
        ../build/lfFocusMaps -i ./data/data -c $V -o ./noreferenceTest -t 1 -r $RANGE -e VAR -m $METHOD -p 32 -y RGB -g $ASPECT -b 5 -z MED -a CLAMP $FAST     
        result=$(../../NIQSV-master/build/exercise ./noreferenceTest/renderImagePostFiltered.png)
        NIQSV=$(grep -oP '(?<=Score: ).*' <<< "$result")

        ffmpeg -y -i ./noreferenceTest/renderImagePostFiltered.png -pix_fmt rgb24 ../../LIQE-main/test.png
        cd ../../LIQE-main/
        result=$(python demo2.py test.png) 
        cd -
        LIQE=$(grep -oP '(?<=quality of).*?(?=as quantified)' <<< "$result")

        echo $SCENE","$METHOD","$FAST","$NIQSV","$LIQE >> $FILE
    done
}

for M in "BF VSET RAND TD HIER DESC PYR"; do
    runTest "buildings" "0.05_0.42" "1.8323" $M
    runTest "cars" "0.15_0.63" "2.003" $M
    runTest "cat" "0.03_0.4" "1.9157" $M
    runTest "class" "0.09_0.58" "2.3807" $M
    runTest "colorful" "0.185_0.4" "1.9058" $M
    runTest "colorless" "0.16_0.41" "1.8084" $M
    runTest "cornell" "0.22_0.39" "1.783" $M
    runTest "diffuse" "0.0_0.52" "2.02762" $M
    runTest "diorama" "0.33_0.85" "1.937" $M
    runTest "face" "0.0_0.44" "1.885" $M
    runTest "greenscreen" "0.18_0.415" "1.9175" $M
    runTest "highfrequency" "0.0_0.45" "1.975" $M
    runTest "largeDepth" "0.0_0.835" "2.0213" $M
    runTest "lowdepth" "0.54_0.63" "2.122" $M
    runTest "lowFrequency" "0.0_0.46" "2.0223" $M
    runTest "macro" "0.1_0.65" "1.9846" $M
    runTest "reflective" "0.0_0.52" "2.02762" $M
    runTest "simpleSetting" "0.43_0.61" "1.8266" $M
    runTest "singleObject" "0.25_0.351" "1.873" $M
    runTest "StanfordBunny" "0.0_0.635" "1.8327" $M
    runTest "text" "0.05_0.43" "1.8658" $M
    runTest "street" "0.16_0.29" "1.816" $M
    runTest "volumetric" "0.0_0.49" "1.89395" $M
    runTest "bonfire" "0.06_0.3" "2.276" $M
    runTest "timelapse" "0.0_0.29" "2.046" $M
    runTest "keying" "0.19_0.53" "1.909" $M
done
