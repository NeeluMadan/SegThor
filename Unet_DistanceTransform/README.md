# SegThor (Unet_DistanceTransform)

Unet_DistanceTransform is a multi-task learning architecture. The main task is pixel-wise segmentation. The auxiliary task is regression of the distance transform values. 


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
python3 main.py --model_name unet_mtl--epochs 40 -b 96 --lr 0.01 \\
--save_dir save_path --data_path ../../../../data --if_auxloss 1 --alpha 0.3


```
#Important Parameter
- --model_name: Indicate the architecture to use i.e. vanilla Unet or multi-task learning architecture

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Possible values: vanilla_unet, unet_mtl

- --if_auxloss: Decide to use the auxiliar loss (regression), --if_auxloss 0 means MTL is not used.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Possible values: 0,1

- --alpha: Decide the weight of regression and segmentation loss

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Possible values: between 0 and 1


## Testing Network

```bash
python3 test.py 

```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

