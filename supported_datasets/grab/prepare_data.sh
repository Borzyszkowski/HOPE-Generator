#!/usr/bin/env bash

mkdir -p ../../_DATA/GRAB
cd ../../_DATA/GRAB
gdown "https://drive.google.com/uc?id=1HTUbQhrN0YtHE5cUPWCFoiDv6sJIg1ny"
unzip GRAB-data.zip
rm GRAB-data.zip
