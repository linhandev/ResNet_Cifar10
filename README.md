# ResNet_Cifar10

## Training

Example usage:

```shell
python train.py \
    --model-name 'resnet_de_resblock' \
    --num-epoch 50 \
    --learning-rate 1e-3 \
    --batch-size 128 \
    --bs-increase-at 30 40 \
    --bs-increase-by 2 2 \ # 2 times then 2 times, eventually 4 times
    --optimizer AdamW \
    --scheduler ReduceLROnPlateau
```

To see all avaliable parameters

```shell
python train.py --help
```

parameters

| Parameter             | Usage                                                                   | Example                                 |
| --------------------- | ----------------------------------------------------------------------- | --------------------------------------- |
| --model-name          | Name of the model to use                                                | resnet_de_resblock                      |
| --num-epoch           | Training epochs                                                         | 50                                      |
| --batch-size          | Initial training batch size. Bs can be adjusted mid-run                 | 128 (For 16GB ram, maximum around 2048) |
| --learning-rate       | Initial learning rate, lr can be scheduled                              | 1e-3                                    |
| --do-aug              | Whether to perform augmentation during training                         | True / False                            |
| --optimizer           | Optimizer to use                                                        | Adam / AdamW / SGD                      |
| --scheduler           | Learning rate scheduler type                                            | ReduceLROnPlateau / PolynomialLR        |
| --bs-increase-at      | At epecified epoch, bs increases                                        | 30 40                                   |
| --bs-increase-by<br/> | How many time does bs increase by, cumulative                           | 2 2                                     |
| --loss                | Loss function to use                                                    | bce / focal                             |
| --model-save-path     | Where to save trained model, training config record and tensorboard log | ./output                                |