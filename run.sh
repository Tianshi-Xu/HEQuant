CUDA_VISIBLE_DEVICES=2 python train.py \
  -c configs/quantized/mbv2_cifar10_300epochs_simple.yml \
  --model cifar10_mobilenetv2_100 /raid/mengli/datasets/ \
  --wq-enable --wq-bitw 1 \
  --initial-checkpoint output/train/20220917-092256-cifar10_mobilenetv2_100-32/model_best.pth.tar \
  --wq-mode TWN \
  --wq-per-channel \
  --use-kd \
  --quant-teacher \
  --teacher cifar10_mobilenetv2_100 \
  --teacher-checkpoint output/train/20220917-092256-cifar10_mobilenetv2_100-32/model_best.pth.tar
  # --initial-checkpoint output/train/20220915-123827-cifar10_mobilenetv2_100-32/model_best.pth.tar

CUDA_VISIBLE_DEVICES=2 python train.py \
  -c configs/quantized/resnet32_cifar10_simple_war_full.yml \
  --model resnet32 /raid/mengli/datasets/ \
  --wq-enable --wq-bitw 8 --wq-mode TWN --wq-per-channel
