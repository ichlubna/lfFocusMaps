#/bin/bash
#Parameters: input rendered folder, output filename, grid row size
TEMP=$(mktemp -d)
ID=0
EXT=""
COUNT=$(ls $1 | wc -l)
MIDDLE=$((COUNT/2))
for FILE in $1/*
do
    EXT=${FILE##*.}
    IDD=$(printf "%05d" $ID)
    cp $FILE $TEMP"/"$IDD"."$EXT
    if [ $ID = $MIDDLE ]
    then
        ffmpeg -i $FILE -vf scale=640:-1 -y $2".jpg"
    fi
    ID=$((ID+1))
done

#ffmpeg -i $TEMP"/%05d."$EXT -c:v libaom-av1 -crf 0 -y $2.mkv
ffmpeg -i $TEMP"/%05d."$EXT -c:v libx264 -y $2.mkv

R=$3
C1=0
C1=$(printf $TEMP"/%05d."$EXT $C1)
C2=$((R-1))
C2=$(printf $TEMP"/%05d."$EXT $C2)
C3=$((COUNT-R))
C3=$(printf $TEMP"/%05d."$EXT $C3)
C4=$((COUNT-1))
C4=$(printf $TEMP"/%05d."$EXT $C4)

convert \( $C1 $C2 +append \) \( $C3 $C4 +append \) -append -resize 640x $2"_thumb.jpg"

rm -rf $TEMP
