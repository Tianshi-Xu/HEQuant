CUDA_VISIBLE_DEVICES=0 python train_quant.py -c configs/quantized/resnet18/cifar100_fix.yml --model ResNet18 /home/xts/code/dataset/cifar100


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_quant.py -c configs/quantized/resnet18/imagenet.yml --model ResNet18 /data/dataset/imagenet/

CUDA_VISIBLE_DEVICES=0,1,6,7 python train_imagenet.py -c configs/quantized/resnet18/imagenet.yml --model ResNet18