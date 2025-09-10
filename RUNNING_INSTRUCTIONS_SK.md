# Instructions for running main.py

## Create environment
```
conda create -n openemma python=3.9 -y
conda activate openemma
```

## Install dependencies
```
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install torch torchvision
```
```
pip install -r requirements.txt
```

## Get Gemini API key

Obtain API key from google AI studio. Replace `<GEMINI_API_KEY>` in `main.py` with the API key.

## Download dataset
Download the full mini dataset from [nuScenes dataset](https://www.nuscenes.org/nuscenes#download).

Extract the files to `datasets/NuScenes/` so that you have the following files:
```
maps  samples  sweeps  v1.0-mini
```

## Run main.py
```
python main.py --model-path gpt --method openemma
```
