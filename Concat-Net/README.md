# SegThor (Concat-Net)

Concat-Net (concatenated network) combined the salient features of 2D and 3D segmentation approach. Four different architectures are proposed.

a) Concat-net with early fusion

b) Concat-net with late fusion

c) Concat-net with auxiliary branch

d) Concat-net with auxiliary branch and skip connections

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install requirement.txt
```
## SegThor Dataset

Can be downloaded from: 
[SegThor_dataset](https://competitions.codalab.org/competitions/21145)

## Training Network

```bash
python3 main.py --model_name early_concat --gpu 0,1 --epochs 32 -b 6 --lr 0.015 \\
--save_dir save_path --data_path ../../../data --with_improvement 0

```
#Important Parameter
- --model_name: Indicate the name of the concat-net architecture to train.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Possible values: early_concat, late_concat, aux_concat, aux_skip_concat

- --with_improvement: If use an improvement term in the training of networks. It indicates if the segmentation is getting better over initial segmentation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Possible values: 0,1


## Testing Network

```bash
python3 test.py --ensemble_method max

```
#Important Parameter 
-  --ensemble_method: Indicate the ensemble technique to use if multiple models are available.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; possible values: None, max, avg

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

