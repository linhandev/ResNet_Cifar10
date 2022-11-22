# ResNet_Cifar10

## Installing Dependencies

- Install [cuda](https://developer.nvidia.com/cuda-downloads) and [cudnn](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html). Newer versions are prefered.

  - It's also possible to get cuda enviroment through [docker](https://hub.docker.com/r/nvidia/cuda/tags). On linux distros not officially supported by Nvidia this can be easier. Choose a tag with cudnn and devel in it, eg: `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04`

- Install pytorch following [official documentation](https://pytorch.org/get-started/locally/)

- Install other dependencies with

  ```shell
  pip install -r requirements.txt
  ```

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

| Parameter             | Usage                                                                    | Example                                 |
| --------------------- | ------------------------------------------------------------------------ | --------------------------------------- |
| --model-name          | Name of the model to use. See full list [here](./models/model_choice.py) | resnet_de_resblock                      |
| --num-epoch           | Training epochs                                                          | 50                                      |
| --batch-size          | Initial training batch size. Bs can be adjusted mid-run                  | 128 (For 16GB ram, maximum around 2048) |
| --learning-rate       | Initial learning rate, lr can be scheduled                               | 1e-3                                    |
| --do-aug              | Whether to perform augmentation during training                          | True / False                            |
| --optimizer           | Optimizer to use                                                         | Adam / AdamW / SGD                      |
| --scheduler           | Learning rate scheduler type                                             | ReduceLROnPlateau / PolynomialLR        |
| --bs-increase-at      | At epecified epoch, bs increases                                         | 30 40                                   |
| --bs-increase-by<br/> | How many time does bs increase by, cumulative                            | 2 2                                     |
| --loss                | Loss function to use                                                     | bce / focal                             |
| --model-save-path     | Where to save trained model, training config record and tensorboard log  | ./output                                |

Use tensorboard to see training updates in real time.

```shell
tensorboard --logdir ./output
```

## Evaluation

The model pt file should be placed at `/{model-save-path}/{model-name}_best.pt`

Example usage:

```shell
python evaluate.py \
    --model-name 'resnet_de_resblock' \
    --model-save-path ./output/resnet_de_resblock-1668939199
```

## Batch scripts

Specify training configs in [tool/configs.csv](tool/configs.csv) and then run

```shell
python tool/batch_train.py
```

The script will search for new configs and run training.

Similarly, run

```shell
python tool/batch_test.py
```
The script will search for training records that haven't performed testing, run test and generate a report at `tool/test_results.csv`

## Code formatting

```shell
pip install pre-commit
pre-commit install --install-hooks # pre-commit will auto run on commit
pre-commit run --all-files
```

## The team

[Haoyang Pei](https://github.com/HaoyangPei), [Samyak Rawlekar](https://samyakr99.github.io/), [linhandev](https://github.com/linhandev)
