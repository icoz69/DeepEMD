#!/bin/sh

#download dataset
#https://drive.google.com/file/d/191cFzwwNTzG_mHUDABF0Nh77cI6pa-qq/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=191cFzwwNTzG_mHUDABF0Nh77cI6pa-qq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=191cFzwwNTzG_mHUDABF0Nh77cI6pa-qq" -O miniimagenet.tar && rm -rf /tmp/cookies.txt

tar -xvf miniimagenet.tar miniimagenet/

