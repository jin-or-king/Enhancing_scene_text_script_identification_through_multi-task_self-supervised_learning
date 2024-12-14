# Enhancing_scene_text_script_identification_through_multi-task_self-supervised_learning
## Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) install = 1.4.0
- torchvision
- CUDA 10.1
- Other dependencies: scipy, pandas, numpy

## training
Our modle is very simple to implement and experiment with.
Our implementation consists in a [main_SJR.py](./main_SJR.py) file from which are imported the dataset definition [src/multicropdataset.py](./src/multicropdataset.py), the model architecture [src/resnet50.py](./src/resnet50.py) and some miscellaneous training utilities [src/utils.py](./src/utils.py).

For example, to train our model, run:
```
python mainSJR.py
--data_path [YOUR CIFAR DATA PATH]
--size_crops 32
--nmb_crops 2
--nmb_prototypes 13
--batch_size 64
--epochs 100
--base_lr 0.06
--final_lr 0.0006
--temperature 0.5
--use_fp16 false
--dump_path checkpoints
--freeze_prototypes_niters 900'
```

# Evaluating models

## Evaluate models: Linear classification on SIW_13
To train a supervised linear classifier on frozen features/weights, run:
```
python eval_linear.py
--root_dir [YOUR SIW13 DATA PATH]
--dump_path ./checkpoints_linear
--pretrained ./checkpoints/[YOUR FILE NAME]
--batch_size 64
--lr 0.03
--epochs 400'
```


