#!/usr/bin/env bash

mkdir -p ./_RESULTS/Downloaded
cd ./_RESULTS/Downloaded

gdown "https://drive.google.com/uc?id=1fpyv5e39F5TQGzi1sCG4wPLBzK91ViG4"
unzip objects_meshes.zip
rm objects_meshes.zip

gdown "https://drive.google.com/uc?id=146DfXp61WYMA23CfK8z4jqytAXlQVX9d"
unzip objects_visualized.zip
rm objects_visualized.zip
