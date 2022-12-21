#!/usr/bin/env bash

mkdir -p ./_BODY_MODELS
cd ./_BODY_MODELS
gdown "https://drive.google.com/uc?id=1T-U3xL63yTl4Sd8tLxOsxCzMDmysTGjw"
unzip BODY_MODELS.zip
rm BODY_MODELS.zip
cp -r ./models/mano/. ./models/
