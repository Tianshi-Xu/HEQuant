# CUDA_VISIBLE_DEVICES=0 python test.py -c \
# configs/quantized/resnet32_w4a4/a4w4.yaml --model resnet32 \
# --data-dir /home/xts/code/dataset/cifar100

# CUDA_VISIBLE_DEVICES=5 python train.py -c \
# configs/quantized/resnet32_w4a4/a2w8.yaml --model resnet32 \
# --data-dir /home/xts/code/dataset/cifar100

CUDA_VISIBLE_DEVICES=1 python train.py -c \
./configs/quantized/mbv2/tiny_a3w3.yaml --model tinyimagenet_mobilenetv2 \
--data-dir /home/xts/code/dataset/tiny-imagenet-200/