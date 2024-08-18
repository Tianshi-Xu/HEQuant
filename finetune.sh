CUDA_VISIBLE_DEVICES=3 python resnet_finetune.py -c \
configs/quantized/resnet32_cifar100_finetune.yml --model resnet32 \
--data-dir ~/datasets --apex-amp  --initial-checkpoint resnet32_a6w8.pth.tar \
--quant-firstlast --wq-per-channel  #--use-kd --teacher wrn40_8_cifar100
