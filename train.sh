CUDA_VISIBLE_DEVICES=3 python train_quant.py -c configs/quantized/resnet18/cifar100_fix.yml --model ResNet18 /home/xts/code/dataset/cifar100


CUDA_VISIBLE_DEVICES=0,1,2,3 python train_quant.py -c configs/quantized/resnet18/imagenet.yml --model ResNet18 /opt/dataset/imagenet/