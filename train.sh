CUDA_VISIBLE_DEVICES=0 python train_quant.py -c configs/quantized/resnet18/cifar100_fix.yml --model ResNet18 /home/xts/code/dataset/cifar100


<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train_quant.py -c configs/quantized/resnet18/imagenet.yml --model ResNet18 /data1/share/imagenet
=======
CUDA_VISIBLE_DEVICES=0,1,6,7 python train_quant.py -c configs/quantized/resnet18/imagenet.yml --model ResNet18 /opt/dataset/imagenet/
>>>>>>> 1c912de60db74f1f6875f7077032d66b075956ce

CUDA_VISIBLE_DEVICES=0,1,6,7 python train_imagenet.py -c configs/quantized/resnet18/imagenet.yml --model ResNet18