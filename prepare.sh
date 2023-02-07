#!/bin/bash

root_path='/notebooks'
repo_name='kohya-trainer-paperspace'

mkdir -p $root_path
if [ ! -d $root_path/$repo_name ]; then
    # clone repo if not exist
    cd $root_path
    git clone https://github.com/sheldonchiu/kohya-trainer-paperspace.git
fi

cd $root_path/$repo_name

echo "Installing Dependencies"
apt-get update && apt-get install -y wget git libgl1 libglib2.0-0
pip install -U pip
pip install torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# pip install protobuf==3.20.3
pip install -r requirements_train.txt
pip install minio python-logging-discord-handler triton==2.0.0.dev20221202
pip install xformers==0.0.16
