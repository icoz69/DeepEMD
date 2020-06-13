#!/bin/sh

#download dataset
#https://drive.google.com/file/d/1Prn7_41NVrZbnePAlSiKjD21Jlz0LKJM/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Prn7_41NVrZbnePAlSiKjD21Jlz0LKJM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Prn7_41NVrZbnePAlSiKjD21Jlz0LKJM" -O deepemd_pretrain_model.tar && rm -rf /tmp/cookies.txt

tar -xvf deepemd_pretrain_model.tar deepemd_pretrain_model/

