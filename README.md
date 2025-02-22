# Dense Depth Map Generation using LiDAR and Stereo Data

## Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Inference](#inference)
4. [Model Overview](#model-overview)
5. [Datasets](#datasets)
6. [Results and Performance](#results-and-performance)
7. [Future Work](#future-work)

## Introduction

This project focuses on generating **dense depth maps** for **autonomous vehicles** using a fusion of **LiDAR** and **Stereo Data**. The method leverages disparity maps from stereo images and combines them with sparse LiDAR data to produce high-quality depth estimations. The goal is to improve depth perception for navigation, object detection, and environmental mapping in self-driving applications.

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch 1.7.0+
- Torchvision
- NumPy
- OpenCV
- KITTI Depth Completion Dataset
- KITTI Stereo Dataset

To install the required Python packages:

```bash
pip install torch torchvision numpy opencv-python
```

## Inference

Dataset is provided in the `kitty_data_tiny` folder, containing raw RGB images and LiDAR depth maps for testing the model.

To test the model, run the following commands:

```bash
chmod +x run_model.sh
./run_model.sh
```

The output depth maps will be stored in the `output_with_stereo_disparities` folder.

- `output_with_RGB/` contains depth maps produced by the baseline model.

## Model Overview

The model consists of the following key components:

- **Spatial Pyramid Pooling Module (SPP)**: Enhances receptive field to extract contextual features.
- **Encoder Network**: Three independent CNN-based encoders process stereo image features and LiDAR data separately.
- **Cost Volume Computation**: Computes stereo correspondences to generate disparity maps.
- **Fusion Module**: Combines LiDAR and stereo image information for depth completion.
- **Decoder Network**: Uses a stacked hourglass architecture to refine the depth map output.
- **Loss Function**: A combination of **Mean Squared Error (MSE)** for depth accuracy and **gradient-based smoothness loss**.

![Model Architecture](README%20Images/Image1.png)  

## Datasets

This model is trained and evaluated on:

1. **KITTI Depth Completion Dataset**

   - Provides LiDAR point clouds and stereo image sequences for depth completion.
   - Contains **42,949 training samples** and **1,000 validation samples**.

2. **KITTI Stereo Dataset**

   - Contains **194 training stereo image pairs** and **195 testing image pairs**.
   - Used for disparity estimation and stereo depth completion.

## Results and Performance

![Comparison with baseline model](README%20Images/Image2.png)  

The model was tested on the KITTI dataset, using standard evaluation metrics:

| Model    | RMSE ↓  | MAE ↓  | iRMSE ↓ | iMAE ↓ |
| -------- | ------- | ------ | ------- | ------ |
| GuideNet | 736.24  | 218.33 | 2.25    | 0.99   |
| Baseline | 792.80  | 225.81 | 2.42    | 0.99   |
| **Ours** | 1548.89 | 493.65 | 5.01    | 1.86   |

While the initial results are not state-of-the-art, qualitative outputs indicate that the model generates consistent and usable depth maps. Performance can be improved with further training and hyperparameter tuning.

## Future Work

- **End-to-End Training**: Currently, disparity and fusion models are trained separately; integrating them could improve performance.
- **Larger Datasets**: Training on **SceneFlow** and full **KITTI datasets** could yield better generalization.
- **Optimized Network Design**: Experimenting with **dilated convolutions** and **attention mechanisms** for better feature extraction.

## Conclusion

This project presents an approach to **fuse stereo and LiDAR data** to generate **dense depth maps** for autonomous vehicles. While the initial performance is promising, further improvements in model architecture and training methodology can enhance depth estimation accuracy.

---

