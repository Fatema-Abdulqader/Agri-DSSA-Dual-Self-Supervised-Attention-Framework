# Agri-DSSA: Dual Self-Supervised Attention Framework

## Overview

This repository contains the MATLAB implementation of **Agri-DSSA**, a dual self-supervised attention framework for multisource crop health analysis using hyperspectral and image-based benchmarks. The framework is designed for both classification and regression tasks in agricultural applications.

## Paper Reference

**Title:** Agri-DSSA: A Dual Self-Supervised Attention Framework for Multisource Crop Health Analysis Using Hyperspectral and Image-Based Benchmarks  
**Author:** Fatema A. Albalooshi

## Features

- **Dual-Attention Mechanism**: Combines spectral and spatial attention for enhanced feature learning
- **Multi-Task Support**: Handles both classification (disease detection) and regression (chlorophyll estimation)
- **Hyperspectral Data Processing**: Specialized preprocessing for hyperspectral imagery
- **Ablation Study**: Comprehensive analysis of model components
- **Attention Visualization**: Tools to interpret model decisions

## Requirements

### Software
- MATLAB R2024a or later
- Deep Learning Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

### Hardware
- Recommended: GPU with CUDA support for faster training
- Minimum: 8GB RAM

## Dataset Support

The code supports the following hyperspectral datasets:
- `Indian_Pines` (with noisy band removal)
- `Pavia_University` 
- `KSC` (with water absorption band removal)

## Usage

### 1. Basic Setup
```matlab
% Set dataset and parameters
datasetName = 'Indian_Pines';
patchSize = 15;
task = 'classification'; % or 'regression'
```

### 2. Data Preprocessing
The code automatically handles:
- Band selection and noise reduction
- Min-Max normalization
- Patch extraction with mirror padding
- Stratified dataset splitting

### 3. Model Training
```matlab
% Training will start automatically after data preparation
% Monitor training progress through the displayed plots
```

### 4. Evaluation
- **Classification**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Regression**: RMSE, MAE, R², Scatter plots

## Model Architecture

### Core Components

1. **Feature Extraction Backbone**
   - 3D convolutional layers for spectral-spatial feature learning
   - 2D convolutional layers for spatial feature refinement

2. **Dual-Attention Module**
   - **Spectral Attention**: Channel-wise feature recalibration
   - **Spatial Attention**: Spatial feature importance weighting

3. **Task-Specific Head**
   - Classification: Softmax output with cross-entropy loss
   - Regression: Linear output with MSE loss

## Key Functions

### Main Functions
- `stratifiedSplit()`: Balanced dataset splitting
- `stopIfNotImproving()`: Early stopping callback
- `calculateClassMetrics()`: Classification performance metrics

### Ablation Study Functions
- `removeSpectralAttention()`: Model variant without spectral attention
- `removeAllAttention()`: Baseline model without attention

## Outputs

### Training
- Training progress plots
- Model validation metrics
- Best model checkpoint

### Evaluation
- Test set performance metrics
- Confusion matrix (classification)
- Regression scatter plots
- Ablation study results

### Visualization
- Spectral attention weights
- Spatial attention maps
- Feature importance maps

## Configuration Options

### Task Selection
```matlab
task = 'classification'; % For disease classification
task = 'regression';    % For chlorophyll estimation
```

### Training Parameters
- Learning rate: 0.001 with step decay
- Batch size: 32
- Maximum epochs: 100
- Early stopping patience: 10 epochs

### Data Parameters
- Patch size: 15×15 pixels
- Train/Val/Test split: 70%/15%/15%
- Band removal based on dataset characteristics

## File Structure

```
Agri-DSSA/
├── Agri_DSSADualSelfSupervised.m              % Main implementation file
├── README.md                                 % This file
└── Results/                                  % Generated outputs
    ├── model_architecture.png
    ├── training_progress.png
    ├── confusion_matrix.png
    └── attention_maps.png
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{albalooshi2024agridssa,
  title={Agri-DSSA: A Dual Self-Supervised Attention Framework for Multisource Crop Health Analysis Using Hyperspectral and Image-Based Benchmarks},
  author={Albalooshi, Fatema A.},
  journal={ },
  year={2025}
}

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or patch size
2. **Slow Training**: Enable GPU acceleration
3. **Dataset Not Found**: Ensure dataset files are in the correct path
4. **NaN Values**: Check data normalization and band selection

### Performance Tips

- Use GPU for training (set `ExecutionEnvironment` to `'gpu'`)
- Adjust patch size based on available memory
- Modify learning rate for different datasets
- Use data augmentation for small datasets

## License

This implementation is for academic research purposes. Please check the original paper for specific licensing details.

## Contact
For questions or issues regarding this implementation, please refer to the original paper or create an issue in the repository.

## Acknowledgments
This implementation is based on the research work "Agri-DSSA: A Dual Self-Supervised Attention Framework for Multisource Crop Health Analysis Using Hyperspectral and Image-Based Benchmarks" by Fatema A. Albalooshi.

