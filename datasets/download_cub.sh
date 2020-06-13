#!/bin/sh

#download dataset
#https://drive.google.com/file/d/1B8jmZin9teye7Lte9ZKsQ3lyMASbxune/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1B8jmZin9teye7Lte9ZKsQ3lyMASbxune' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1B8jmZin9teye7Lte9ZKsQ3lyMASbxune" -O cub.tar && rm -rf /tmp/cookies.txt

tar -xvf cub.tar cub/

