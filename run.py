import os

cmd = 'python mainSJR.py --data_path [YOUR CIFAR DATA PATH] --size_crops 32 --nmb_crops 2 --nmb_prototypes 30 --batch_size 64 --epochs 100 --base_lr 0.06 --final_lr 0.0006 --temperature 0.5 --use_fp16 false --dump_path checkpoints --freeze_prototypes_niters 900'
os.system(cmd)
cmd = 'python eval_linear.py --root_dir [YOUR SIW13 DATA PATH] --dump_path ./checkpoints_linear --pretrained ./checkpoints/[YOUR FILE NAME] --batch_size 64 --lr 0.03 --epochs 400'
os.system(cmd)
