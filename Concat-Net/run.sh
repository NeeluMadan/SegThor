python3 main.py --model_name early_concat --gpu 0,1 --epochs 32 -b 6 --lr 0.015 --save_dir save_path --data_path ../../../data --with_improvement 0
python3 main.py --model_name late_concat --gpu 0,1 --epochs 32 -b 6 --lr 0.015 --save_dir save_path --data_path ../../../data --with_improvement 0 
python3 main.py --model_name aux_concat --gpu 0,1 --epochs 32 -b 4 --lr 0.015 --save_dir save_path --data_path ../../../data --with_improvement 0
python3 main.py --model_name aux_skip_concat --gpu 0,1 --epochs 32 -b 4 --lr 0.015 --save_dir save_path --data_path ../../../data --with_improvement 0


