#!/usr/bin/env bash

mkdir -p ./_RESULTS/Downloaded
cd ./_RESULTS/Downloaded

gdown "https://drive.google.com/uc?id=1kFK1bb4F_rGJ0bkEuSGdVjLTecqU__8o"
unzip 100_objects_meshes.zip
rm 100_objects_meshes.zip

gdown "https://drive.google.com/uc?id=1idFs92EU3JswTtUPopBmisSKWaFnw6bl"
unzip 100_objects_visualized.zip
rm 100_objects_visualized.zip
