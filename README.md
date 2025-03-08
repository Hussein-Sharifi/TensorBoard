# FashionMNIST Classification with CNN & TensorBoard

## Overview
This project implements a **Convolutional Neural Network (CNN)** for classifying images from the **FashionMNIST dataset**. The key focus is on **visualizing training performance and model predictions using TensorBoard**.

## Features
- **Convolutional Neural Network** (2 Conv layers, 3 Fully Connected layers)
- **TensorBoard Integration** for:
  - Loss tracking
  - Model architecture visualization
  - Image embeddings (Projector)
  - Prediction confidence graphs
  - Precision-Recall curves

## Repository Contents
- `Visualization.py` – CNN model and TensorBoard logging.
- `README.md` – Project documentation.
- `Notes/` – Additional notes or related files.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/fashion-mnist-cnn-tensorboard.git
   cd fashion-mnist-cnn-tensorboard
   ```

2. Install dependencies:
   ```sh
   pip install torch torchvision tensorboard matplotlib numpy
   ```

## Usage
Run the training script:
```sh
python Visualization.py
```

This will:

   ```sh
   1. Train a CNN model on FashionMNIST.
   2. Log training progress to TensorBoard.
   3. Save model graphs, loss metrics, and embeddings.
   ```

## Model Architecture
Convolutional Layers:
-Conv2d(1 → 6, kernel_size=5), MaxPool
-Conv2d(6 → 16, kernel_size=5), MaxPool

Fully Connected Layers:
-Linear(16*4*4 → 120), ReLU
-Linear(120 → 84), ReLU
-Linear(84 → 10) (Softmax over 10 classes)

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

## TensorBoard visualization
Start TensorBoard:
   ```sh
   tensorboard --logdir=runs
   ```
Then open http://localhost:6006/ in your browser.

## TensorBoard Features:
-Scalars	     | Tracks training loss over time
-Graphs	       | Visualizes the CNN architecture
-Images    	   | Displays sample input images
-Embeddings	   | Projects image features in 2D/3D space
-PR Curves     | Precision-recall visualization for each class

## Training Parameters
-Optimizer     | SGD (momentum=0.9)
-Loss Function | CrossEntropyLoss
Batch Size     | 4
Learning Rate  | 0.001
Epochs         | 1 (modifiable)

## References
This model is based on [this PyTorch tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html).
