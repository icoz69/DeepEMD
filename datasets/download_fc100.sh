#!/bin/sh

#download dataset
#https://drive.google.com/file/d/1nEh3O2RJ5zTbWj7luyQCNcklpX0_5KlS/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nEh3O2RJ5zTbWj7luyQCNcklpX0_5KlS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nEh3O2RJ5zTbWj7luyQCNcklpX0_5KlS" -O FC100.tar && rm -rf /tmp/cookies.txt

tar -xvf FC100.tar FC100/

