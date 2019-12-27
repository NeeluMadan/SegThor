## Run unet with MTL
python3 main.py -b 96 --model_name unet_mtl --gpu 1 --epochs 10 --lr 0.085 --save_dir save_path --data_path ../../../../data --if_auxloss 1 --alpha 0.3


## Run unet without MTL
# python3 main.py -b 96 --model_name vanilla_unet --gpu 1 --epochs 10 --lr 0.02 --save_dir save_path --data_path ../../../../data --if_auxloss 0 --alpha 0.0
#python3 main.py --model_name vanilla_unet --epochs 2 -b 96 --lr 0.01 --save_dir save_path --data_path ../../../../data --if_auxloss 0 --alpha 0.0
