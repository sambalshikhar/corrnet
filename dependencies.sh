#!/bin/bash

mkdir /content/nfs/
sudo apt-get install nfs-common
sudo mount 13.68.129.235:/home/ubuntu/extended_nfs /content/nfs;

echo "nfs mounted" 
git clone https://github.com/facebookresearch/fastText.git
pip install ./fastText
pip install annoy
pip install -U deep_translator
cp $1 .

echo "Depencies installed"