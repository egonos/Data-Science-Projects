# Breast Cancer Detection Project

## Overview

This project focuses on breast cancer detection using deep learning models. The dataset consists of histopathology image patches derived from whole slide images (WSIs). The primary task is to classify each patch as either IDC positive (Invasive Ductal Carcinoma) or IDC negative.

The motivation behind this work is twofold:

1. To identify the best performing classifier for this task.
2. To evaluate **MobileNetâ€™s claim** that depthwise separable convolutions combined with width and resolution multipliers produce lightweight models with minimal accuracy loss ([Howard et al., 2017](https://arxiv.org/abs/1704.04861)), by comparing its performance and computational cost against other well-known architectures.

Each notebook corresponds to a single model and walks through its training, evaluation, and results.

- **Part 1:** MobileNet + Classification Head
- **Part 2:** A Custom CNN architectureÂ + Classification Head
- **Part 3:** VGG16 +Â Classification Head
- **Part 4:** ResNetÂ +Â Classification Head
- **Part 5:** Model Comparison

## Dataset

The [dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) used in this study focuses on Invasive Ductal Carcinoma (IDC), the most common subtype of breast cancer. It was originally derived from 162 whole mount slide images of breast cancer specimens scanned at 40x magnification. From these slides, 277,524 image patches of size 50Ã—50 pixels were extracted, comprising 198,738 IDC-negative and 78,786 IDC-positive samples. Each image patch is labeled according to its class (0: non-IDC, 1: IDC) and includes positional metadata in its filename (patient ID, xâ€“y coordinates, and class).Â 

## Dataset Preparation

\*\*Dataset Split\*\*

The dataset was split into three folds: 70% training, 15% validation, and 15% testing. Since the dataset is imbalanced, stratification was employed.&#x20;



\*\*Rescaling\*\*

Different models required different normalization strategies:

- **Custom CNN:** Input images were scaled to the [0,1] range by dividing pixel values by 255.
- **Pre-trained models (MobileNet, VGG16, ResNet):** Each model used its respective built-in preprocessing function from `keras.applications` (e.g., `mobilenet_v2.preprocess_input`, `vgg16.preprocess_input`, `resnet.preprocess_input`) to stay compatible with pretrained weights.



\*\*Efficient Data Loading\*\*

The input pipeline was built with `tf.data.Dataset` for throughput and reproducibility. Training data was **cached** and **prefetched** (with `tf.data.AUTOTUNE`) to hide I/O latency and keep the GPU fed. Parsing/augmentation used `num_parallel_calls=tf.data.AUTOTUNE` where applicable.

## Models Implemented

1. **Custom CNN:** The Custom CNN model is designed as a simple yet effective baseline, consisting of three conventional convolutional blocks:

- `Conv2D â†’ BatchNorm â†’ ReLU â†’ MaxPooling` (Conv Block)

In each block:

- **Conv2D:** Extracts spatial features. The kernel size decreases across blocks (e.g., 9Ã—9 â†’ 6Ã—6 â†’ 3Ã—3), while the number of filters increases.
- **BatchNorm:** Normalizes intermediate activations, helping to stabilize training and prevent gradients from exploding or vanishing.
- **ReLU:** Introduces non-linearity and reduces the risk of saturation.
- **MaxPooling:** Reduces spatial resolution, increases efficiency, and provides  translation invariance.

This architecture serves as a benchmark to compare against more complex, pretrained models.

The convolutional blocks are followed by a **shared classification head**, used consistently across all models for fair comparison:

- `Dense(10) â†’ ReLU â†’ Dense(1) â†’ Sigmoid` (Classification Head)

Thus, the full Custom CNN pipeline can be summarized as:

- `3 Ã— (Conv Block) â†’ Classification Head`

2. **Transfer Learning Models:** For VGG16, MobileNet, and ResNet, the pretrained backbone is connected to the identical classification head:

- `(Transfer Learning Backbone) â†’ Classification Head`

All transfer learning backbones were frozen during training. In addition, convolution layers were configured with `padding='same'` and `stride=1` throughout training.

## Evaluation Metrics

Since the dataset is imbalanced, the accuracy score is not reliable. Instead, precision, recall, F1-score, and ROC-AUC can be considered. Among these, **ROC-AUC** was selected as the main evaluation metric because it provides robust model comparisons under different class imbalance scenarios ([Boughorbel et al., 2024](https://www.sciencedirect.com/science/article/pii/S2666389924001090)). For the loss function, the binary cross entropy was used.



## Model Training

1. **Class Weights**

   To address the class imbalance, weights were applied to emphasize the minority class. The weights were calculated using the balanced formula:

```
class_weight_i = n_samples / (n_classes * n_i)
```

where `n_i` is the number of samples in class *i*.

2. **Callbacks**
   - **Early Stopping:** Training was conducted with a relatively high maximum epoch limit (300), but early stopping was applied if the validation loss did not improve for 5 consecutive epochs.Â The best-performing model weights (based on validation metrics) were saved and restored at the end of training to ensure final evaluation used the optimal parameters.
   - **Learning Rate Scheduling:** When validation performance plateaued for two rounds, the learning rate was reduced by half.



## Results

Some of the raw learning curves appeared noisy. To better visualize the training dynamics, exponential moving averages were applied for smoothing. With this adjustment, all models (except the CNN baseline) show steadily improving validation performance over time. The complete set of plots can be found in the *Learning Curves and Performance Plots* folder.

![Model comparison 1](<Learning Curves and Performance Plots/model_comparison1.png>)

![Model comparison 2](<Learning Curves and Performance Plots/model_comparison2.png>)

From the figures above, it can be concluded that the ResNet model achieves the best overall performance by a small margin (Also it has the best test set F1-Score which is 0.744). However, when considering parameter count and multiplyâ€“accumulate (MAC) operations, MobileNet stands out as the most efficient model.

## Conclusion & Further Improvements

This project demonstrated how different deep learning architectures (Custom CNN, VGG16, ResNet, and MobileNet) perform on breast cancer histopathology image classification. The comparison showed that ResNet achieved the best performance overall, while MobileNet offered the best trade-off between accuracy and efficiency.

**Future Directions:**

- Use a more complex classification head to explore whether richer decision boundaries improve performance.
- Perform systematic hyperparameter optimization (e.g., using HyperBand or Bayesian optimization) to extract maximum performance from each model.
- Extend experiments to additional lightweight architectures such as EfficientNet.
- Investigate ensemble methods that combine strengths of multiple models.
- Explore deployment as a clinical decision-support web application.



## References

1. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications.Â *arXiv preprint arXiv:1704.04861*.
2. Richardson, E., Trevizani, R., Greenbaum, J. A., Carter, H., Nielsen, M., & Peters, B. (2024). The receiver operating characteristic curve accurately assesses imbalanced datasets.Â *Patterns*,Â *5*(6).


