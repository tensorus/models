# tensorus-models

## Overview
Welcome to `tensorus-models`! This repository is a comprehensive extension to the core [Tensorus](https://github.com/tensorus/tensorus) library, offering a rich and diverse collection of pre-built machine learning models. Designed to accelerate your research and development, `tensorus-models` provides ready-to-use implementations spanning classical machine learning, deep learning (including computer vision, NLP, generative models, GNNs, and RL), time series analysis, and more.

All models are accessible under the `tensorus.models` namespace and are designed to seamlessly integrate with the utilities and infrastructure of the main `tensorus` library, such as `tensorus.tensor_storage`. Whether you're looking for a robust baseline, a component for a larger system, or a starting point for novel research, `tensorus-models` aims to be your go-to resource for a wide array of model architectures.


## Key Features

*   **Extensive Model Zoo:** Access a vast collection of models across diverse machine learning domains, from classical algorithms to state-of-the-art deep learning architectures.
*   **Tensorus Ecosystem Integration:** Seamlessly integrates with the core `tensorus` library, leveraging its utilities for tensor storage, data handling, and more.
*   **Ready-to-Use Implementations:** Save development time with pre-built, tested models that can be easily imported and used in your projects.
*   **Broad Categorization:** Models are organized into clear categories, making it easy to find the right architecture for your specific task.
*   **Foundation for Research:** Provides robust baseline implementations that can be extended or adapted for novel research and experimentation.

## Installation

Install the package in editable mode while in the repository directory:

```bash
pip install -e .
```

When published on PyPI you will also be able to install it with:

```bash
pip install tensorus-models
```

### Example

```python
from tensorus.models import LinearRegressionModel
```

## Available Models

This package provides a wide variety of machine learning models. They are broadly categorized as follows:

**I. Classical Machine Learning**

*   **Regression:**
    *   `CatBoostRegressorModel`
    *   `DecisionTreeRegressorModel`
    *   `ElasticNetRegressionModel`
    *   `GradientBoostingRegressorModel`
    *   `LassoRegressionModel`
    *   `LightGBMRegressorModel`
    *   `LinearRegressionModel`
    *   `PolynomialRegressionModel`
    *   `PoissonRegressorModel`
    *   `RandomForestRegressorModel`
    *   `RidgeRegressionModel`
    *   `SVRModel` (Support Vector Regression)
    *   `XGBoostRegressorModel`
*   **Classification:**
    *   `CatBoostClassifierModel`
    *   `DecisionTreeClassifierModel`
    *   `GaussianNBClassifierModel` (Gaussian Naive Bayes)
    *   `GradientBoostingClassifierModel`
    *   `KNNClassifierModel` (K-Nearest Neighbors)
    *   `LDAClassifierModel` (Linear Discriminant Analysis)
    *   `LightGBMClassifierModel`
    *   `LogisticRegressionModel`
    *   `MLPClassifierModel` (Multi-layer Perceptron)
    *   `QDAClassifierModel` (Quadratic Discriminant Analysis)
    *   `RandomForestClassifierModel`
    *   `SVMClassifierModel` (Support Vector Machine)
    *   `XGBoostClassifierModel`
*   **Clustering:**
    *   `AgglomerativeClusteringModel`
    *   `DBSCANClusteringModel`
    *   `GaussianMixtureModel`
    *   `KMeansClusteringModel`
*   **Dimensionality Reduction & Embedding:**
    *   `CCAModel` (Canonical Correlation Analysis)
    *   `FactorAnalysisModel`
    *   `PCADecompositionModel` (Principal Component Analysis)
    *   `TSNEEmbeddingModel` (t-SNE)
    *   `UMAPEmbeddingModel`
*   **Anomaly Detection:**
    *   `IsolationForestModel`
    *   `OneClassSVMModel`

**II. Deep Learning - Computer Vision**

*   **Image Classification:**
    *   `AlexNetModel`
    *   `EfficientNetModel`
    *   `LeNetModel`
    *   `MobileNetModel`
    *   `ResNetModel`
    *   `VGGModel`
    *   `VisionTransformerModel` (ViT)
*   **Object Detection:**
    *   `FasterRCNNModel`
    *   `YOLOv5Detector`
*   **Image Segmentation:**
    *   `UNetSegmentationModel`

**III. Deep Learning - Natural Language Processing**

*   **Language Models & Sequence Classification:**
    *   `BERTModel`
    *   `GPTModel`
    *   `GRUClassifierModel`
    *   `LSTMClassifierModel`
    *   `T5Model`
    *   `TransformerModel`
    *   `LargeLanguageModelWrapper`
*   **Word Embeddings:**
    *   `GloVeModel`
    *   `Word2VecModel`
*   **Named Entity Recognition:**
    *   `NamedEntityRecognitionModel`

**IV. Deep Learning - Generative Models**

*   `DiffusionModel`
*   `FlowBasedModel`
*   `GANModel` (Generative Adversarial Network)
*   `VAEModel` (Variational Autoencoder)

**V. Deep Learning - Graph Neural Networks (GNNs)**

*   `GATClassifierModel` (Graph Attention Network)
*   `GCNClassifierModel` (Graph Convolutional Network)

**VI. Deep Learning - Reinforcement Learning (RL)**

*   `A2CModel` (Advantage Actor-Critic)
*   `DQNModel` (Deep Q-Network)
*   `PPOModel` (Proximal Policy Optimization)
*   `QLearningModel`
    *   `SACModel` (Soft Actor-Critic)
*   `TRPOModel` (Trust Region Policy Optimization)

**VII. Time Series Analysis**

*   `ARIMAModel`
*   `ExponentialSmoothingModel`
*   `GARCHModel`
*   `SARIMAModel`

**VIII. Recommender Systems**

*   `CollaborativeFilteringModel`
    *   `MatrixFactorizationModel`
    *   `NeuralCollaborativeFilteringModel`

**IX. Statistical Models & Advanced Analytics**

*   `AnovaModel` (Analysis of Variance)
*   `CoxPHModel` (Cox Proportional Hazards Survival Model)
*   `ManovaModel` (Multivariate Analysis of Variance)
*   `MixedEffectsModel` (Linear Mixed-Effects Models)
*   `StructuralEquationModel`

**X. Specialized & Other Models**

*   **Federated Learning:**
    *   `FedAvgModel`
*   **Meta-Learning / Ensemble Methods:**
    *   `MixtureOfExpertsModel`
    *   `StackedGeneralizationModel` (Stacking)
    *   `StackedRBMClassifierModel` (Stacked Restricted Boltzmann Machines)
*   **Semi-Supervised Learning:**
    *   `LabelPropagationModel`
    *   `SelfTrainingClassifierModel`
*   **Emerging/Research Areas:**
    *   `MultimodalFoundationModel`
    *   `NeuroSymbolicModel`
    *   `PhysicsInformedNNModel`

## Running Tests

The tests require the core `tensorus` package so that `tensorus.tensor_storage` can be imported. Either clone the main repository and install it, or install it from PyPI:

```bash
# Option 1: clone Tensorus
git clone https://github.com/tensorus/tensorus
pip install -e tensorus

# Option 2: install from PyPI
pip install tensorus
```

After installing `tensorus` and the requirements for this repository, run:

```bash
pip install -r requirements.txt
pytest
```
