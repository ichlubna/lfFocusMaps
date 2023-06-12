#!/bin/bash

runTest() {
    SCENE=$1
    RANGE=$2
    ASPECT=$3
    rm -rf ./data/*
    python ./renameLFImages.py ./dataset/nonanim/${SCENE}/001/ ./data/ 0 0 15 15 0 1 27
    python measure.py ./data/data/ ./data/reference/ $RANGE 8 8 $ASPECT > ./results/$SCENE.txt 2>&1 
}

runTest "buildings" "0.05_0.42" "1.8323"
runTest "cars" "0.15_0.63" "2.003"
runTest "cat" "0.03_0.4" "1.9157"
runTest "class" "0.09_0.58" "2.3807"
runTest "colorful" "0.185_0.4" "1.9058"
runTest "colorless" "0.16_0.41" "1.8084"
runTest "cornell" "0.22_0.39" "1.783"
runTest "diffuse" "0.0_0.52" "2.02762"
runTest "diorama" "0.33_0.85" "1.937"
runTest "face" "0.0_0.44" "1.885"
runTest "greenscreen" "0.18_0.415" "1.9175"
runTest "highfrequency" "0.0_0.45" "1.975"
runTest "largeDepth" "0.0_0.835" "2.0213"
runTest "lowdepth" "0.54_0.63" "2.122"
runTest "lowFrequency" "0.0_0.46" "2.0223"
runTest "macro" "0.1_0.65" "1.9846"
runTest "reflective" "0.0_0.52" "2.02762"
runTest "simpleSetting" "0.43_0.61" "1.8266"
runTest "singleObject" "0.25_0.351" "1.873"
runTest "StanfordBunny" "0.0_0.635" "1.8327"
runTest "text" "0.05_0.43" "1.8658"
runTest "volumetric" "0.0_0.49" "1.89395"
runTest "bonfire" "0.06_0.3" "2.276"
runTest "timelapse" "0.0_0.29" "2.046"
runTest "keying" "0.19_0.53" "1.909"
