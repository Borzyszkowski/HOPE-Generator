#!/usr/bin/env bash

mkdir -p ./_SOURCE_DATA/GRAB
cd ./_SOURCE_DATA/GRAB
gdown "https://drive.google.com/uc?id=1HTUbQhrN0YtHE5cUPWCFoiDv6sJIg1ny"
unzip GRAB_data.zip
rm GRAB_data.zip
