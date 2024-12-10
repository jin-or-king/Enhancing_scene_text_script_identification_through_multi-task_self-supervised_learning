import os

cmd = 'python mainSwav+Combined+Total+low.py --data_path E:\data\C10 --size_crops 32 --nmb_crops 2 --nmb_prototypes 30 --batch_size 64 --epochs 100 --base_lr 0.06 --final_lr 0.0006 --temperature 0.5 --use_fp16 false --dump_path checkpoints --freeze_prototypes_niters 900'
os.system(cmd)
cmd = 'python eval_linear_Swav+Combined+Total+low.py --root_dir E:\data\data\SIW\dataset\SIW-13 --dump_path ./checkpoints_linear --pretrained ./checkpoints/checkpoint_low34_all_75_1_0.6_0.8.pth.tar --batch_size 64 --lr 0.03 --epochs 402'
os.system(cmd)