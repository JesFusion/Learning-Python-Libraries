#!/bin/bash


cd /home/jesfusion/Documents/ml/ML-Learning-Repository/Docker/Docker_Folders/

echo "Enter folder name: "

read folder_name

mkdir $folder_name

cd $folder_name

touch Dockerfile
touch file.py
touch build.sh

echo "Done creating folder '$folder_name' with files"






