## Quantization results summary

### How to run experiments

```bash
python sweep.py -c configs/quantized/cct_7-3x1_cifar10_300epochs_ar.yml --model vit_6_4_32 /data/home/menglifrl/dataset --aq-enable --aq-mode LSQ --aq-bitw 2 --mixup 0.8 --lr 75e-5 --weight-decay 0.05 --use-kd --teacher vit_6_4_32 --teacher-checkpoint output/train/20220510-202140-vit_6_4_32-32/model_best.pth.tar 
```

### Key finding

- Two-step quantization (activation --> weight) achieves better accuracy compared to one-step quantization.

- Activation quantization/ternarization is more challenging compared to weight quantization/ternarization 

- With proper knowledge distillation (e.g., the floating point model or more accuate models), the accuracy of weight quantized model is close to the floating point model (w/o KD)

- For weight quantized networks, using a more capable teacher (cct-7-3x1-32) improves the accuracy compared to small teacher (vit-6-4-32) from 92.56 to 92.99. In contrast, for activation quantized networks, using a more capable teacher does not benefit the accuracy (90.83 vs 90.26)

- When further quantizing the attention, we see further accuracy degradation, however, changing the attention weight quantization to (0, 1, 2) instead of (-1, 0, 1) increases the accuracy.

- Multi-step quantization helps with improving model accuracy:
    - Use cct-7-3x1-32 (18376, 96.12%) to distill vit-6-4-32 (24269, 94.34%) compared to baseline vit-6-4-32 (14289, 93.03%)
    - Use vit-6-4-32 (24269, 94.34%) to distill 4b vit-6-4-32 (39923, 93.85%) compared to kd (25878, 93.02%) with baseline vit-6-4-32 (14289, 93.03%)
    - Use 4b vit-6-4-32 to distill 2b vit-6-4-32 (39181, 91.22%) compared to kd (24259, 90.83%) with baseline vit-6-4-32 (14289, 93.03%)

- KD with both logits and intermediate embedding helps
    - KD with distilled vit-6-4-32 (24269, 94.34%), logits only: 28353, 91.11%
    - KD with distilled vit-6-4-32 (24269, 94.34%), logits + last embedding: 39174, 92.32%
    - KD with distilled vit-6-4-32 (24269, 94.34%), logits + average all embedding: 

<!-- # - How to further improve the network accuracy?
#     - Multi-step KD
#     - Attention/activation KD -->


## Final Experiments

| Network           | Tricks                                                                            | Accuracy@310 epochs   | Run ID | Output Dir      |
|-------------------|-----------------------------------------------------------------------------------|-----------------------|--------|-----------------|
| vit-6-4-32        | lr 0.00075, wd 0.05                                                               |                       | 51847  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, kd (cct-7-3x1-32)                                     |                       | 51848  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, wq 2b, sym                                            |                       | 51851  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, wq 1b, sym                                            |                       | 51861  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, wq 2b, asym                                           |                       | 51852  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, aq 2b, sym, linear                                    |                       | 51853  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, aq 2b, asym, linear                                   |                       | 51854  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, aq 2b, sym, linear, use relu                          |                       | 51855  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, aq 2b, asym, linear, use relu                         |                       | 51856  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, aq 2b, sym, linear + attn                             |                       | 51857  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, aq 2b, asym, linear + attn                            |                       | 51858  |                 |
| vit-6-4-32        | lr 0.00075, wd 0.05, no aa, aq 2b, asym, linear + attn, use relu                  |                       | 51859  |                 |



## Experiments


| Network           | Tricks                                                                            | Accuracy@310 epochs   | Run ID   | Output Dir      |
|-------------------|-----------------------------------------------------------------------------------|-----------------------|----------|-----------------|
| vit-6-4-32 fp     | Baseline setting                                                                  | 93.03                 | 14289    | 20220504-183710 |
| vit-6-4-32 fp     | Baseline setting, lr 0.00075, wd 0.05, use kd                                     | 94.34                 | 24270    | 20220510-202459 |
| vit-6-4-32 fp     | Baseline setting, lr 0.00075, wd 0.05, no aa, use kd                              | 94.54                 | 24269    | 20220510-202140 |
| vit-6-4-32 fp     | Baseline setting, lr 0.00075, wd 0.05, no aa, use kd (cct-7-3x1-32), relu         | 93.90                 | 39931    | 20220517-182740 |
| vit-6-4-32 fp     | Baseline setting, bn                                                              | nan                   | 14603    | 20220504-183710 |
| vit-6-4-32-sine fp| Baseline setting                                                                  | 92.86                 | 14290    | 20220504-184019 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.06                                                 | 87.57                 | 14496    | 20220504-212356 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.01                                                 | 86.21                 | 14926    | 20220505-070050 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.005, reprob 0.25, no aa                            | 88.16                 | 14937    | 20220505-095822 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.06, reprob 0.25, mixup 0                           | 87.18                 | 15141    | 20220505-172853 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.06, reprob 0.0, no aa, mixup 0.8                   | 89.16                 | 15139    | 20220505-172543 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.06, reprob 0.25, no aa, mixup 0.8                  | 89.14                 | 15138    | 20220505-172421 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.06, reprob 0.25, no aa, mixup 0                    | 88.83                 | 14949    | 20220505-075702 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, mixup 0        | 88.93                 | 15142    | 20220505-173034 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa                 | 89.48                 | 15133    | 20220505-235208 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, no smooth      | 89.40                 | 17115    | 20220506-061507 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00125, wd 0.03, reprob 0.25, no aa, use kd         | 90.02                 | 17466    | 20220506-083212 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00125, wd 0.03, reprob 0.25, no aa, use kd, no smooth       | 90.02                 | 17465    | 20220506-083036 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.001, wd 0.04, reprob 0.25, no aa, use kd, no smooth         | 90.83                 | 17464    | 20220506-083013 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.001, wd 0.04, reprob 0.25, no aa, use kd           | 90.83                 | 17463    | 20220506-082909 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd         | 91.11                 | 17462    | 20220506-082831 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (vit-6-4-32)     | 90.83                 | 24259    | 20220510-201515 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (vit-6-4-32), q linear     | 92.13                 | 28342    | 20220513-025228 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (vit-6-4-32), q proj       | 92.85                 | 28337    | 20220513-025037 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32)     | 91.11                 | 28353    | 20220513-025544 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.001, wd 0.04, reprob 0.25, no aa, use kd (4b vit-6-4-32, 25878)       | 90.92                 | 39185    | 20220517-041915 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (4b vit-6-4-32, 25878)     | 91.22                 | 39181    | 20220517-041046 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32, mse, alpha 1.0)     | 92.23            | 39172    | 20220517-013419 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32, mse, alpha 2.0)     | 92.32            | 39173    | 20220517-013434 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32, mse, alpha 4.0)     | 92.22            | 39174    | 20220517-013703 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32, mse all layers, alpha 2.0)     | 91.43            | 42402    | 20220518-073129 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (4b vit-6-4-32, mse, alpha 2.0), relu      | 92.61            | 42841    | 20220518-182347 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32, mse, alpha 2.0)             | 92.26         | 46456    | 20220519-211858 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32, mse, alpha 2.0), relu       | 92.56         | 46457    | 20220519-212015 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (4b vit-6-4-32, mse, alpha 2.0), relu, asym q      | 93.41         | 43045    | 20220518-221958 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (4b vit-6-4-32, mse, alpha 2.0)            | 92.07            | 45211    | 20220519-163356 |
| vit-6-4-32 aq     | Baseline setting, act 4b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (vit-6-4-32, 14289)                        | 93.02                 | 25878    | 20220511-205709 |
| vit-6-4-32 aq     | Baseline setting, act 4b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32, 24269)              | 93.85                 | 39923    | 20220517-174117 |
| vit-6-4-32 aq     | Baseline setting, act 4b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32, 24269, mse, alpha 2.0) | 94.21              | 39919    | 20220517-172527 |
| vit-6-4-32 aq     | Baseline setting, act 4b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (distilled vit-6-4-32, 24269, mse all layers, alpha 2.0) | 93.98              | 42386    | 20220518-062914 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (cct-7-3x1-32)   | 90.36                 | 24222    | 20220510-201223 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd, attn q | 90.22                 | 18172    | 20220506-181332 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd, attn q | 89.87                 | 25911    | 20220511-211426 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd, attn q, pos attn | 90.53                 | 25997    | 20220511-224949 |
| vit-6-4-32 aq     | Baseline setting, act 4b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd, attn q, pos attn | 93.24                 | 26001    | 20220511-225114 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd, learnable bias         | 90.09                 | 18188    | 20220506-182556 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, learnable bias, two head 0.5   | 90.43                 | 18374    | 20220506-212617 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, learnable bias, two head 0.8   | 90.75                 | 18372    | 20220506-212513 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.00075, wd 0.05, reprob 0.25, no aa, learnable bias, two head 1.0   | 90.88                 | 18373    | 20220506-212536 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.001, wd 0.03, reprob 0.25, no aa                   | 88.37                 | 16138    | 20220505-235439 |
| vit-6-4-32 aq     | Baseline setting, act 2b, lr 0.0015, wd 0.02, reprob 0.25, no aa                  | 85.81                 | 16140    | 20220505-235453 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.06, reprob 0.25, no aa, mixup 0.8, agc 0.5         | 89.07                 | 16166    | 20220505-000826 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.06, reprob 0.25, no aa, mixup 0.8, norm 0.5        | 89.11                 | 16161    | 20220505-000714 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.06, reprob 0.25, no aa, mixup 0.8, norm 1.0        | 89.23                 | 16160    | 20220505-000620 |
| vit-6-4-32 aq     | Baseline setting, act 2b, wd 0.06, reprob 0.25, no aa, mixup 0.8, agc 0.1         | 89.20                 | 16165    | 20220505-000817 |
| vit-6-4-32 waq    | Baseline setting, act 2b, w 2b                                                    | 85.33                 | 14510    | 20220504-214949 |
| vit-6-4-32 wq     | Baseline setting, w 2b                                                            | 91.42                 | 14918    | 20220505-065058 |
| vit-6-4-32 wq     | Baseline setting, w 2b, no aa, mixup 0                                            | 91.61                 | 15143    | 20220505-173518 |
| vit-6-4-32 wq     | Baseline setting, w 2b, no aa, mixup 0.8, reprob 0.25                             | 92.31                 | 16201    | 20220506-002107 |
| vit-6-4-32 wq     | Baseline setting, w 2b, no aa, mixup 0.8, reprob 0.25, kd (cct-7-3x1-32)          | 92.99                 | 24221    | 20220510-200849 |
| vit-6-4-32 wq     | Baseline setting, w 2b, no aa, mixup 0.8, reprob 0.25, kd (vit-6-4-32)            | 92.56                 | 24220    | 20220510-200748 |
| vit-6-4-32 wq     | Baseline setting, w 2b, no aa, mixup 0.8, per channel                             | 90.94                 | 16202    | 20220505-002131 |
| vit-6-4-32 wq     | Baseline setting, w 4b, lr 0.00075, wd 0.05, reprob 0.25, no aa, use kd (vit-6-4-32)     | 93.24                 | 25883    | 20220511-205841 |
| vit-6-4-32 wq-aq  | Baseline setting, w 2b, act 2b, no aa, mixup 0.8, reprob 0.25, kd (vit-6-4-32), wq (24220), lr 0.00075 wd 0.05            | 91.04                 | 25861    | 20220511-203916 |
| vit-6-4-32 aq-wq  | Baseline setting, w 2b, act 2b, no aa, mixup 0.8, reprob 0.25, kd (vit-6-4-32), wq (24259), lr 0.00075 wd 0.05            | 91.38                 | 25863    | 20220511-204505 |
|-------------------|-----------------------------------------------------------------------------------|-----------------------|----------|-----------------|
| cct-7-3x1-32      | Baseline setting                                                                  | 96.12                 | 18376    | 20220506-213137 |
| cct-7-3x2-32      | Baseline setting, lr 0.00075, wd 0.05, mixup 0.8, no aa                           | 94.52                 | 39935    | 20220517-185313 |

## Cifar100 Experiments

|-------------------|-----------------------------------------------------------------------------------|-----------------------|----------|-----------------|
| vit-6-4-32 fp     | Baseline setting                                                                  | 74.14                 | 46466    | 20220519-213916 |
