CUDA_VISIBLE_DEVICES=1 python ILP.py -c configs/quantized/resnet18/imagenet.yml --model ResNet18 /home/xts/code/dataset/cifar100


CUDA_VISIBLE_DEVICES=3 python ILP.py -c configs/quantized/resnet18/imagenet_ILP.yml --model ResNet18 /data/dataset/imagenet/
