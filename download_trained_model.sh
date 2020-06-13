#!/bin/sh

#download dataset
#https://drive.google.com/file/d/1lGcNHMRnBrjODDmt647RzMJ5cLCd4pmv/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lGcNHMRnBrjODDmt647RzMJ5cLCd4pmv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lGcNHMRnBrjODDmt647RzMJ5cLCd4pmv" -O deepemd_trained_model.tar && rm -rf /tmp/cookies.txt

tar -xvf deepemd_trained_model.tar deepemd_trained_model/

