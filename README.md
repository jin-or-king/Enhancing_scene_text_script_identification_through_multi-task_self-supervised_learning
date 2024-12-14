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

For example, to train SwAV baseline on a single node with 8 gpus for 400 epochs, run:
```
python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
--data_path /path/to/imagenet/train \
--epochs 400 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15
```

# Evaluating models

## Evaluate models: Linear classification on ImageNet
To train a supervised linear classifier on frozen features/weights on a single node with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py \
--data_path /path/to/imagenet \
--pretrained /path/to/checkpoints/swav_800ep_pretrain.pth.tar
```
The resulting linear classifier can be downloaded [here](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar).

