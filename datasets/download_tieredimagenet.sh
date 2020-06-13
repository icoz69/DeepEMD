#!/bin/sh

#download dataset
#https://drive.google.com/file/d/1ANczVwnI1BDHIF65TgulaGALFnXBvRfs/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ANczVwnI1BDHIF65TgulaGALFnXBvRfs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ANczVwnI1BDHIF65TgulaGALFnXBvRfs" -O tieredimagenet.tar && rm -rf /tmp/cookies.txt

tar -xvf tieredimagenet.tar tieredimagenet/

