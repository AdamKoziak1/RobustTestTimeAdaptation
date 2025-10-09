# RobustTestTimeAdaptation

This codebase is mainly based on [TSD](https://github.com/SakurajimaMaiii/TSD) 
## Setup
Use a virtual environment:
```
conda create -n tta python=3.13.5
conda activate tta
```
Alternatively, venv:
```
python3.13 -m venv tta
source tta/bin/activate
```

Then, install the requirements:
```
pip install -r requirements.txt
```

## Dataset
Download the datasets:  
[PACS](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd)  
[OfficeHome](https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC)  
[VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)  
Organize them as follows:
```
|datasets
  |office-home
    |Art
    |Clipart
    |Product
    |RealWorld
  |PACS
    |art_painting
    |cartoon
    |photo
    |sketch
  |VLCS
    |Caltech101
    |LabelMe
    |SUN09
    |VOC2007
```

## Train source model
Please use `train.py` to train the source model. For example:
```bash
python train.py --dataset PACS \
                --data_dir your_data_dir \
                --opt_type Adam \
                --lr 5e-5 \
                --max_epoch 50
```
Change `--dataset PACS` for other datasets, such as `office-home`, `VLCS`.  
## Adversarial Data Generation
```bash
python generate_adv_data.py --dataset PACS \
                            --eps 8 \
                            --alpha_adv 0.5 \
                            --steps 20 \
```

## Test time adaptation
```bash
python unsupervise_adapt.py --dataset PACS \
                            --data_dir your_data_dir \
                            --adapt_alg Tent \ 
                            --lr 1e-4 \
                            --fft_input_keep_ratio 0.8 \
                            --fft_input_alpha 0.8 \
                            --fft_feat_keep_ratio 0.8 \
                            --fft_feat_alpha 0.8
```
Change `--adapt_alg TSD` to use different methods of test time adaptation, e.g. `T3A`, `Tent`.  
Change `--fft*` to adjust the parameters relating to the FFT modules.

Ensure that you configure the default arguments in these files to match your directory structure.

## ✉️ Contact
Please contact adamkoziak@cmail.carleton.ca
