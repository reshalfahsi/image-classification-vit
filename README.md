# Image Classification With Vision Transformer

 <div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/image-classification-vit/blob/master/Image_Classification_With_Vision_Transformer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
 </div>


Vision Transformer, or ViT, tries to perform image recognition in a sequence modeling way. By dividing the image into patches and feeding them to the revered model in language modeling, a.k.a. Transformer, ViT shows that over-reliance on the spatial assumption is not obligatory. However, a study shows that giving more cues about spatial information, i.e., subjecting consecutive convolutions to the image before funneling it to the Transformer, aids the ViT in learning better. Since ViT employs the Transformer block, we can easily receive the attention map _explaining_ what the network sees. In this project, we will be using the CIFAR-100 dataset to examine ViT performance. Here, the validation set is fixed to be the same as the test set of CIFAR-100. Online data augmentations, e.g., RandAugment, CutMix, and MixUp, are utilized during training. The learning rate is adjusted by following the triangular cyclical policy.


## Experiment

Experience the journey of training, testing, and inference image classification with ViT by jumping to this [notebook](https://github.com/reshalfahsi/image-classification-vit/blob/master/Image_Classification_With_Vision_Transformer.ipynb).


## Result

## Quantitative Result

Here are the quantitative results of ViT performance:

Test Metric  | Score
------------ | -------------
Loss         | 1.353
Top1-Acc.    | 64.92%
Top5-Acc.    | 87.29%


## Accuracy and Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/image-classification-vit/blob/master/assets/acc_curve.png" alt="acc_curve" > <br /> Accuracy curves of ViT on the CIFAR-100 test set. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/image-classification-vit/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> Loss curves of ViT on the CIFAR-100 test set. </p>


## Qualitative Result

The predictions and the corresponding attention maps are served in this collated image.

<p align="center"> <img src="https://github.com/reshalfahsi/image-classification-vit/blob/master/assets/qualitative.png" alt="qualitative" > <br /> Several prediction results of ViT and their attention map. </p>


## Credit

- [An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)
- [TorchVision's ViT](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)
- [Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/)
- [Train a Vision Transformer on small datasets](https://keras.io/examples/vision/vit_small_ds/)
- [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [The CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/pdf/1909.13719.pdf)
- [RandAugment for Image Classification for Improved Robustness](https://keras.io/examples/vision/randaugment/)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899.pdf)
- [CutMix data augmentation for image classification](https://keras.io/examples/vision/cutmix/)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf)
- [MixUp augmentation for image classification](https://keras.io/examples/vision/mixup/)
- [Early Convolutions Help Transformers See Better](https://arxiv.org/pdf/2106.14881.pdf)
- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)
- [How to use CutMix and MixUp](https://pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
