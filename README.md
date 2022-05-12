# E2EGI
E2EGI: End-to-End Gradient Inversion in Federated Learning
(Gradient Leakage in Federated Learning)

## Abstract:

A large amount of healthcare data is produced every day as more and more Internet Medical Things are utilized, such as wearable swatches and bracelets. Mining valuable information from the data distributed at the owners is important and useful, but it is challenging to preserve data privacy when the data is transferred to central servers. Federated learning is believed to be a promising approach to handle the problem in recent years. Recent studies have demonstrated that Gradient Inversion Attack can reconstruct the input data by leaked gradients, without accessing the raw data. However, previous work can only achieve gradient inversion attacks in very limited scenarios, such as the label repetition rate of the target sample being low and batch sizes being smaller than 48. In this paper, we propose a new method E2EGI. Compared to the state-of-the-art method, E2EGI's Minimum Loss Combinatorial algorithm can realize reconstructed samples with higher similarity, and the Distributed Gradient Inversion algorithm can implement gradient inversion attacks with batch sizes of 8 to 256 on deep network models (such as ResNet-50) and Imagenet datasets(224X224), which means malicious cilents can reconstruct input samples from aggregated gradients. We also propose a new Label Reconstruction algorithm that relies only on the gradient information of the target model, which can achieve a label reconstruction accuracy of 81\% with a label repetition rate of 96\%, a 27\% improvement over the state-of-the-art method. 


---

## Requirements

## Get model gradient

## Perform E2EGI


