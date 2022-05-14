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


## Perform E2EGI

## Reproduce our results

### 1. Training task with batch size (8)

#### 1.1 Get model gradients of the target sample
Run the following code to get checkpoint with the model gradient of the target samples (model: trained ResNet-50 with mocov2, parameters can be downloaded at this  URL [moco](https://github.com/facebookresearch/moco), target sample: imagenet-train, batch size 8)
```
python training.py -b 8 --gpu 0 --pretrained [mocov2-folder/moco_v2_800ep_pretrain.pth.tar] --data [imagenet-folder with train]
```
Obtain the checkpoint by executing the above command, and its path is ./train/checkpoint/idtest_resnet50_moco_v2_800ep_pretrain.pth.tar_imagenet_b8_i0-checkpoint.pth.tar

When the batch size is small, gradient inversion can be performed using only a single GPU, and three modes of operation are described next. You can choose one of the commands 1.2, 1.3, or 1.4 to run.

#### 1.2 basic gradient inversion
Do not use group consistency regularization, and need to provide regularization weights lr, TV, BN (optional)
```
python run.py --gpu 0 --exact-bn --n-seed 1 --grad-sign --input-boxed --min-grads-loss --epochs 24000 --lr 0.1046 --TV 0.0114 --BN 0.0357 --pseudo-label-init known --metric --one-to-one-similarity --GInfoR --checkpoint ./train/checkpoint/idtest_resnet50_moco_v2_800ep_pretrain.pth.tar_imagenet_b8_i0-checkpoint.pth.tar

```

#### 1.3 basic gradient inversion with Hyperparameter Search
The hyperparameter search engine will search for hyperparameters lr, TV, BN. For demonstration purposes, simulation-checkpoint in the command can be the same as checkpoint (test only). Note that the path must use an absolute path.
```
python run.py --gpu 0 --exact-bn --n-seed 1 --grad-sign --input-boxed --min-grads-loss --epochs 24000 --pseudo-label-init known --metric --one-to-one-similarity --GInfoR --checkpoint [E2EGI-folder/train/checkpoint/idtest_resnet50_moco_v2_800ep_pretrain.pth.tar_imagenet_b8_i0-checkpoint.pth.tar] --superparameters-search --lr-tune --TV-tune --BN-tune --simulation-checkpoint [path, generated by own data]
```

#### 1.4 gradient inversion with Minimum Loss Combinatorial Optimization (MLCO)
MLCO fuses multiple groups (n-seed: 8) of pseudo-samples to form group consistency regularization to obtain globally better-reconstructed samples (one of our main contributions). We can get the appropriate hyperparameters through 1.3, and then execute MLCO.
```
python run.py --gpu 0 --exact-bn --grad-sign --input-boxed --min-grads-loss --epochs 24000 --lr 0.1046 --TV 0.0114 --BN 0.0357 --pseudo-label-init known --metric --one-to-one-similarity --GInfoR --checkpoint ./train/checkpoint/idtest_resnet50_moco_v2_800ep_pretrain.pth.tar_imagenet_b8_i0-checkpoint.pth.tar --MinCombine --n-seed 8 
```

### 2. Training task with batch size (256)

#### 2.1 Get model gradients of the target sample
The difference here is that to obtain the model gradients corresponding to 256 batches of input samples, multiple (8) GPUs are required to run. 
```
python training.py -b 256 --pretrained [mocov2-folder/moco_v2_800ep_pretrain.pth.tar] --data [imagenet-folder with train] --dist-url tcp://127.0.0.1:10011 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 
```
Obtain the checkpoint by executing the above command, and its path is ./train/checkpoint/idtest_resnet50_moco_v2_800ep_pretrain.pth.tar_imagenet_b256_i0-checkpoint.pth.tar

#### 2.2 Distributed gradient inversion
The above 1.1 and 1.3 can both use distributed gradient inversion (multi-GPU execution) to support gradient inversion tasks with larger batch sizes. It is only necessary to remove the gpu setting and add relevant commands after the command: `--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:10034 --dist-backend nccl --multiprocessing-distributed`. For example, applying distributed gradient inversion in MLCO:
```
python run.py --exact-bn --grad-sign --input-boxed --min-grads-loss --epochs 24000 --lr 0.1046 --TV 0.0114 --BN 0.0357 --pseudo-label-init known --metric --one-to-one-similarity --checkpoint ./train/checkpoint/idtest_resnet50_moco_v2_800ep_pretrain.pth.tar_imagenet_b256_i0-checkpoint.pth.tar --MinCombine --n-seed 8 --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:10034 --dist-backend nccl --multiprocessing-distributed
```

## Custom

Choice of different models and input samples [model: resnet18, input samples: [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (idx: 0, batch size: 8)]
```
python training.py -a resnet18  -b 8 --gpu 0 --pretrained [file path with resnet18.pth] --data-name celeba --data [celeba-folder] --target-idx 0 --outlayer-state kaiming_uniform --results ./train/checkpoint 
```
More options can be seen in the parser description in the file training.py. After the above command is run, the corresponding checkpoint will be obtained, and the gradient inversion algorithm can be executed. See 1.2, 1.3, 1.4, 2.2.
