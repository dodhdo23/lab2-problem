# Lab 2: CIFAR-10 Image Classification

This project implements and compares several image classification models on the CIFAR-10 dataset. The experiments include a linear classifier, MLPs, custom ConvNets, and ResNet50 variants.

## Project Structure

```text
lab2-problem/
├── configs/
│   ├── linear.yaml
│   ├── mlp_512.yaml
│   ├── mlp_2048.yaml
│   ├── convnet_v1.yaml
│   ├── convnet_v2.yaml
│   ├── convnet_v3_reg.yaml
│   ├── convnet_v3_aug.yaml
│   ├── resnet50_scratch.yaml
│   └── resnet50_finetune.yaml
├── src/
├── train.py
├── requirements.txt
└── README.md
```

## Environment

The experiments were run on Google Colab with an NVIDIA A100 GPU.

```bash
pip install -r requirements.txt
```

## How to Run

Each experiment can be executed with the corresponding config file.

```bash
python train.py --config configs/linear.yaml
python train.py --config configs/mlp_512.yaml
python train.py --config configs/mlp_2048.yaml
python train.py --config configs/convnet_v1.yaml
python train.py --config configs/convnet_v2.yaml
python train.py --config configs/convnet_v3_reg.yaml
python train.py --config configs/convnet_v3_aug.yaml
python train.py --config configs/resnet50_scratch.yaml
python train.py --config configs/resnet50_finetune.yaml
```

In Colab, the following option was used to reduce dataloader overhead:

```bash
python train.py --config configs/convnet_v3_aug.yaml --opts dataset.num_workers=2
```

## Results

The table reports the best validation performance for each model. The validation set is used as the CIFAR-10 test evaluation result in this assignment setting.

| Model | Best Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Trainable Params |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 27 | 1.5707 | 0.4692 | 1.7468 | 0.4018 | 30,730 |
| MLP-512 | 62 | 0.0008 | 1.0000 | 5.6357 | 0.5507 | 1,841,162 |
| MLP-2048 | 75 | 0.0036 | 0.9989 | 6.6012 | 0.5598 | 10,510,346 |
| ConvNet v1 | 34 | 0.0003 | 1.0000 | 1.4655 | 0.8033 | 1,556,106 |
| ConvNet v2 | 86 | 0.0000 | 1.0000 | 0.9830 | 0.8825 | 29,619,370 |
| ConvNet v3 Reg | 87 | 0.5022 | 0.9998 | 0.8304 | 0.8957 | 29,619,370 |
| ConvNet v3 Aug | 82 | 0.5669 | 0.9723 | 0.6245 | 0.9543 | 29,619,370 |
| ResNet50 Scratch | 83 | 0.0000 | 1.0000 | 0.9682 | 0.8466 | 23,528,522 |
| ResNet50 Finetune | 18 | 0.2732 | 0.9074 | 0.1586 | 0.9482 | 23,528,522 |

## Analysis

The linear model achieved the lowest validation accuracy, showing that a simple linear classifier is not sufficient for CIFAR-10 image classification. The MLP models improved over the linear baseline, but they showed strong overfitting. Both MLP models reached almost perfect training accuracy while validation accuracy stayed around 0.55.

The ConvNet models performed much better because convolutional layers can extract spatial image features. ConvNet v1 reached 0.8033 validation accuracy, and ConvNet v2 improved this result to 0.8825 by using a larger and deeper architecture.

ConvNet v3 Reg improved generalization by applying regularization techniques such as dropout, stochastic depth, label smoothing, and weight decay. ConvNet v3 Aug achieved the best overall result with 0.9543 validation accuracy. This shows that data augmentation was highly effective for reducing overfitting and improving generalization.

ResNet50 fine-tuning also achieved strong performance with 0.9482 validation accuracy. Compared with ResNet50 trained from scratch, fine-tuning performed better because the model could reuse pretrained ImageNet features.

## Conclusion

Among all experiments, ConvNet v3 Aug showed the best performance with 0.9543 validation accuracy. The results confirm that deeper convolutional architectures, regularization, and data augmentation are important for improving CIFAR-10 classification performance.
