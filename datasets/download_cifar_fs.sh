#!/bin/sh

#download dataset
#https://drive.google.com/file/d/1Lq2USoQmbFgCFJlGx3huFSfjqwtxMCL8/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Lq2USoQmbFgCFJlGx3huFSfjqwtxMCL8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Lq2USoQmbFgCFJlGx3huFSfjqwtxMCL8" -O cifar_fs.tar && rm -rf /tmp/cookies.txt

tar -xvf cifar_fs.tar cifar_fs/

