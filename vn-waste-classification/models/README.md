# Models Documentation

This directory contains documentation about the models used in the VN Waste Classification project. Below is a brief description of each model architecture implemented in this project, along with their intended use cases.

## Model Architectures

### ResNet18
- **Description**: A deep residual network with 18 layers, ResNet18 is designed to address the vanishing gradient problem by using skip connections.
- **Use Case**: Suitable for image classification tasks with moderate complexity. It is efficient and provides a good balance between performance and computational cost.

### ResNet50
- **Description**: An extension of ResNet18, ResNet50 has 50 layers and utilizes bottleneck layers to improve performance while maintaining a manageable number of parameters.
- **Use Case**: Ideal for more complex image classification tasks where deeper networks can capture more intricate features.

### MobileNetV2
- **Description**: A lightweight model designed for mobile and edge devices, MobileNetV2 uses depthwise separable convolutions to reduce the number of parameters.
- **Use Case**: Best suited for applications where computational resources are limited, such as mobile applications.

### DenseNet121
- **Description**: DenseNet121 connects each layer to every other layer in a feed-forward fashion, which helps in feature reuse and reduces the number of parameters.
- **Use Case**: Effective for tasks requiring high accuracy and where overfitting is a concern due to limited data.

### EfficientNet-B0
- **Description**: A family of models that scale up in a balanced way, EfficientNet-B0 is the baseline model that achieves high accuracy with fewer parameters.
- **Use Case**: Suitable for a wide range of image classification tasks, particularly when efficiency and accuracy are both critical.

### VGG16
- **Description**: A classic convolutional neural network architecture known for its simplicity and depth, VGG16 consists of 16 layers with small receptive fields.
- **Use Case**: Often used as a benchmark for image classification tasks, it is effective but computationally expensive.

## Training and Evaluation
Each model is trained separately using the training scripts provided in the `src` directory. The training process includes saving the trained models and logging the training metrics for further analysis.

## Visualization
Training metrics such as loss and accuracy are visualized using the plotting functions in the `src/plot.py` file, allowing for an easy assessment of model performance over epochs. 

For detailed instructions on how to train and evaluate these models, please refer to the `notebooks/train_and_evaluate.ipynb` file.