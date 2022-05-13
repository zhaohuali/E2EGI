# E2EGI (updating)
E2EGI: End-to-End Gradient Inversion in Federated Learning
(Gradient Leakage in Federated Learning)

## Abstract:

A large amount of healthcare data is produced every day as more and more Internet Medical Things are utilized, such as wearable swatches and bracelets. Mining valuable information from the data distributed at the owners is important and useful, but it is challenging to preserve data privacy when the data is transferred to central servers. Federated learning is believed to be a promising approach to handle the problem in recent years. Recent studies have demonstrated that Gradient Inversion Attack can reconstruct the input data by leaked gradients, without accessing the raw data. However, previous work can only achieve gradient inversion attacks in very limited scenarios, such as the label repetition rate of the target sample being low and batch sizes being smaller than 48. In this paper, we propose a new method E2EGI. Compared to the state-of-the-art method, E2EGI's Minimum Loss Combinatorial algorithm can realize reconstructed samples with higher similarity, and the Distributed Gradient Inversion algorithm can implement gradient inversion attacks with batch sizes of 8 to 256 on deep network models (such as ResNet-50) and Imagenet datasets(224X224), which means malicious cilents can reconstruct input samples from aggregated gradients. We also propose a new Label Reconstruction algorithm that relies only on the gradient information of the target model, which can achieve a label reconstruction accuracy of 81\% with a label repetition rate of 96\%, a 27\% improvement over the state-of-the-art method. 


---

## Requirements

```
pytorch=1.10.1
torchvision=0.11.2
apex: https://github.com/NVIDIA/apex

```

## Reproduce our results

### Training task with batch size (8)

Run the following code to get the model gradient of the target sample (model: trained ResNet-50 with mocov2, parameters can be downloaded at this  URL [moco](https://github.com/facebookresearch/moco), target sample: imagenet-train, batch size 8)
```
python training.py -b 8 --gpu 0 --pretrained [mocov2-folder/moco_v2_800ep_pretrain.pth.tar] --results ./train/checkpoint --data-backup ./train
```


### Training task with batch size (256)
The difference here is that to obtain the model gradients corresponding to 256 batches of input samples, multiple (8) GPUs are required to run.
```
python training.py -b 256 --pretrained [mocov2-folder/moco_v2_800ep_pretrain.pth.tar] --results ./train/checkpoint --data-backup ./train --dist-url tcp://127.0.0.1:10011 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0
```


## Custom

Choice of different models and input samples [model: resnet18, input samples: celebA (idx: 0, batch size: 8)]
```
python training.py -a resnet18  -b 8 --gpu 0 --pretrained [file path with resnet18.pth] --data-name celeba --data [celeba-folder] --target-idx 0 --results ./train/checkpoint 
```
More options can be seen in the parser description in the file training.py