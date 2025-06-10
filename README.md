# Tensorus Models

## Overview
Welcome to `tensorus-models`! This repository is a comprehensive extension to the core [Tensorus](https://github.com/tensorus/tensorus) library, offering a rich and diverse collection of pre-built machine learning models. Designed to accelerate your research and development, `tensorus-models` provides ready-to-use implementations spanning classical machine learning, deep learning (including computer vision, NLP, generative models, GNNs, and RL), time series analysis, and more.

All models are accessible under the `tensorus.models` namespace and are designed to seamlessly integrate with the utilities and infrastructure of the main `tensorus` library, such as `tensorus.tensor_storage`. Whether you're looking for a robust baseline, a component for a larger system, or a starting point for novel research, `tensorus-models` aims to be your go-to resource for a wide array of model architectures.


## Key Features

*   **Extensive Model Zoo:** Access a vast collection of models across diverse machine learning domains, from classical algorithms to state-of-the-art deep learning architectures.
*   **Tensorus Ecosystem Integration:** Seamlessly integrates with the core `tensorus` library, leveraging its utilities for tensor storage, data handling, and more.
*   **Ready-to-Use Implementations:** Save development time with pre-built, tested models that can be easily imported and used in your projects.
*   **Broad Categorization:** Models are organized into clear categories, making it easy to find the right architecture for your specific task.
*   **Foundation for Research:** Provides robust baseline implementations that can be extended or adapted for novel research and experimentation.


## Quick Start

This section provides a few examples to get you started with `tensorus-models`. First, ensure you have the package installed:

```bash
pip install tensorus-models
```

Below are examples for some of the commonly used model types. These examples use minimal data and epochs for brevity. For real-world applications, ensure your data is appropriately preprocessed and models are trained for a sufficient number of iterations.

### 1. XGBoostRegressorModel (Classical ML - Regression)

This example demonstrates training an XGBoost regression model.

```python
import numpy as np
from tensorus.models import XGBoostRegressorModel

# 1. Create dummy data
print("1. Creating dummy data...")
X_train = np.random.rand(100, 5)  # 100 samples, 5 features
y_train = np.random.rand(100)     # 100 target values
X_test = np.random.rand(20, 5)    # 20 samples for prediction
print(f"   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"   X_test shape: {X_test.shape}")

# 2. Initialize the XGBoostRegressorModel
print("\n2. Initializing XGBoostRegressorModel...")
# For GPU usage: model = XGBoostRegressorModel(use_gpu=True, tree_method="gpu_hist")
model = XGBoostRegressorModel(n_estimators=10, random_state=42) # Using few estimators for speed
print(f"   Model initialized: {model}")

# 3. Fit the model
print("\n3. Fitting the model...")
model.fit(X_train, y_train)
print("   Model fitting complete.")

# 4. Make predictions
print("\n4. Making predictions...")
predictions = model.predict(X_test)
print(f"   Predictions shape: {predictions.shape}")
print(f"   First 5 predictions: {predictions[:5]}")

# 5. Save and load the model (optional)
print("\n5. Saving the model...")
model_path = "xgboost_regressor.json"
model.save(model_path)
print(f"   Model saved to {model_path}")
print("\n6. Loading the model...")
loaded_model = XGBoostRegressorModel()
loaded_model.load(model_path)
print("   Model loaded successfully.")
loaded_predictions = loaded_model.predict(X_test)
assert np.array_equal(predictions, loaded_predictions), "Predictions from loaded model do not match!"
print("   Predictions from loaded model verified.")

print("\nQuick Start example for XGBoostRegressorModel finished.")
```

### 2. ARIMAModel (Time Series Analysis)

This example shows how to use the ARIMA model for time series forecasting.

```python
import numpy as np
import pandas as pd
from tensorus.models import ARIMAModel

# 1. Create dummy time series data
print("1. Creating dummy time series data...")
data = np.random.randn(100) + np.arange(100) * 0.1  # Trend + noise
time_series = pd.Series(data)
print(f"   Time series length: {len(time_series)}")
print(f"   Last 5 data points: \n{time_series.tail()}")

# 2. Initialize the ARIMAModel
print("\n2. Initializing ARIMAModel...")
# Example order (p,d,q); optimal order depends on data characteristics.
model = ARIMAModel(order=(1, 1, 1))
print(f"   Model initialized with order (1,1,1): {model}")

# 3. Fit the model
print("\n3. Fitting the model...")
model.fit(time_series)
print("   Model fitting complete.")

# 4. Make predictions (forecast)
print("\n4. Making predictions (forecasting next 10 steps)...")
forecast_steps = 10
predictions = model.predict(steps=forecast_steps)
print(f"   Forecasted {forecast_steps} steps: {predictions}")

# 5. Save and load the model (optional)
print("\n5. Saving the model...")
model_path = "arima_model.joblib"
model.save(model_path)
print(f"   Model saved to {model_path}")
print("\n6. Loading the model...")
loaded_model = ARIMAModel(order=(1,1,1))
loaded_model.load(model_path)
print("   Model loaded successfully.")
loaded_predictions = loaded_model.predict(steps=forecast_steps)
assert np.array_equal(predictions, loaded_predictions), "Predictions from loaded model do not match!"
print("   Predictions from loaded model verified.")

print("\nQuick Start example for ARIMAModel finished.")
```

### 3. TransformerModel (NLP - Generic Sequence-to-Sequence)

This example demonstrates the generic `TransformerModel` for a toy sequence-to-sequence task. This model can be a base for tasks like translation or summarization.

```python
import numpy as np
import torch
from tensorus.models import TransformerModel

# 1. Define vocabulary and create dummy sequence data
print("1. Creating dummy tokenized data for a sequence-to-sequence task...")
source_vocab_size = 10 # Small vocabulary for source
target_vocab_size = 10 # Small vocabulary for target
batch_size = 2
seq_len = 5

X_train_tokens = np.random.randint(0, source_vocab_size, size=(batch_size, seq_len))
# Target for .fit() should be the complete target sequence.
# For teacher forcing, model internally shifts target for decoder input.
y_train_tokens = np.random.randint(0, target_vocab_size, size=(batch_size, seq_len + 1)) # e.g., +1 for EOS or shifted
print(f"   Source vocab size: {source_vocab_size}, Target vocab size: {target_vocab_size}")
print(f"   X_train_tokens (batch_size, seq_len):\n{X_train_tokens}")
print(f"   y_train_tokens (batch_size, seq_len+1):\n{y_train_tokens}")

# 2. Initialize the TransformerModel
print("\n2. Initializing TransformerModel...")
model = TransformerModel(
    input_dim=source_vocab_size, output_dim=target_vocab_size,
    model_dim=16, num_heads=2, num_encoder_layers=1, num_decoder_layers=1,
    dim_feedforward=32, epochs=2 # Minimal epochs for example
)
print(f"   Model initialized.")

# 3. Fit the model
print("\n3. Fitting the model (minimal example)...")
model.fit(X_train_tokens, y_train_tokens)
print("   Model fitting complete.")

# 4. Make predictions
print("\n4. Making predictions (generating sequences)...")
X_test_tokens = X_train_tokens[:1]
print(f"   Input for prediction (X_test_tokens):\n{X_test_tokens}")
predicted_sequence_tokens = model.predict(X_test_tokens, max_len=seq_len + 2)
# Assuming predicted_sequence_tokens is a torch tensor
print(f"   Predicted sequence tokens (first batch item):\n{predicted_sequence_tokens[0].cpu().numpy()}")


# 5. Save and load the model (optional)
print("\n5. Saving the model...")
model_path = "transformer_model.pth"
model.save(model_path)
print(f"   Model saved to {model_path}")
print("\n6. Loading the model...")
loaded_model = TransformerModel(
    input_dim=source_vocab_size, output_dim=target_vocab_size,
    model_dim=16, num_heads=2, num_encoder_layers=1, num_decoder_layers=1,
    dim_feedforward=32, epochs=2
)
loaded_model.load(model_path)
print("   Model loaded successfully.")
loaded_predictions = loaded_model.predict(X_test_tokens, max_len=seq_len + 2)
assert torch.equal(predicted_sequence_tokens, loaded_predictions), "Predictions from loaded model!"
print("   Predictions from loaded model verified.")

print("\nQuick Start example for TransformerModel finished.")
```

### 4. DQNModel (Reinforcement Learning)

This example shows basic usage of the `DQNModel` with a very simple dummy environment.

```python
import numpy as np
import torch
import random # For dummy environment
from tensorus.models import DQNModel

# 0. Define a very simple dummy environment
class DummyEnv:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.max_steps = 20 # Short episodes for demo
        self.current_step = 0
    def reset(self):
        self.current_state = np.random.rand(self.state_dim).astype(np.float32)
        self.current_step = 0
        return self.current_state
    def step(self, action):
        self.current_step += 1
        # Simple reward: 1 if action is "correct" (e.g. matches a property of state)
        reward = 1.0 if action == np.argmax(self.current_state) % self.action_dim else -0.1
        next_state = np.random.rand(self.state_dim).astype(np.float32)
        done = self.current_step >= self.max_steps
        self.current_state = next_state
        return next_state, reward, done, {} # {} is for info dict (often empty)
    def __repr__(self):
        return f"<DummyEnv state_dim={self.state_dim} action_dim={self.action_dim}>"

print("Quick Start for Reinforcement Learning (DQNModel)")

# 1. Initialize the dummy environment
print("\n1. Initializing Dummy Environment...")
state_dimension = 4
action_dimension = 2
dummy_env = DummyEnv(state_dim=state_dimension, action_dim=action_dimension)
print(f"   Dummy Environment: {dummy_env}")

# 2. Initialize the DQNModel
print("\n2. Initializing DQNModel...")
dqn_model = DQNModel(
    state_dim=state_dimension, action_dim=action_dimension,
    hidden_size=32, lr=0.001, gamma=0.9, epsilon=0.1, # Epsilon for exploration
    batch_size=8, buffer_size=100, target_update=5 # Small values for quick demo
)
print(f"   Model initialized.")

# 3. Fit the model (train the agent)
print("\n3. Fitting the model (training for a few episodes)...")
num_episodes = 2 # Minimal episodes for this example
dqn_model.fit(dummy_env, episodes=num_episodes)
print(f"   Model fitting complete after {num_episodes} episodes.")

# 4. Make predictions (select actions)
print("\n4. Making predictions (selecting actions)...")
# Create a new dummy state or use one from env.reset()
dummy_state = np.random.rand(state_dimension).astype(np.float32)
action = dqn_model.predict(dummy_state)
print(f"   For dummy state (first 2 features): {dummy_state[:2]}...")
print(f"   Predicted action: {action}")

# 5. Save and load the model (optional)
print("\n5. Saving the model...")
model_path = "dqn_model.pth"
dqn_model.save(model_path)
print(f"   Model saved to {model_path}")

print("\n6. Loading the model...")
# Re-initialize with same parameters before loading state dict
loaded_dqn_model = DQNModel(state_dim=state_dimension, action_dim=action_dimension, hidden_size=32)
loaded_dqn_model.load(model_path)
print("   Model loaded successfully.")

# Verify loaded model can predict
loaded_action = loaded_dqn_model.predict(dummy_state)
print(f"   Loaded model predicted action for dummy_state: {loaded_action}")
# Note: For RL, exact prediction match after loading isn't always guaranteed if training was short
# or if the predict method itself had stochasticity (not typical for DQN's final action selection).

print("\nQuick Start example for DQNModel finished.")
```

For more detailed examples and advanced usage, please refer to the specific model documentation (if available) or the example scripts in the repository.

## Installation

Get started with `tensorus-models` by installing it using pip:

```bash
pip install tensorus-models
```

**Requirements:**
*   Python >= 3.8
*   Core Dependencies: `torch`, `torchvision`, `transformers`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `numpy`, `joblib`.
    *   *Note:* Some models may have additional specific dependencies (e.g., `pandas` and `statsmodels` for `ARIMAModel`). These are typically imported by the model files themselves and may need to be installed separately if not already present in your environment. The core dependencies listed above are automatically installed with `tensorus-models`.

### For Developers (Editable Install)

If you wish to contribute to `tensorus-models` or need an editable installation for development purposes, you can clone the repository and install it locally:

```bash
git clone https://github.com/tensorus/tensorus-models.git # Please replace with your actual repository URL if different!
cd tensorus-models
pip install -e .
```
To set up a development environment with all necessary dependencies for testing and contribution, please refer to the `requirements.txt` file and the "Running Tests" section.
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


## Contributing

Contributions to `tensorus-models` are welcome! Whether it's adding new models, improving existing ones, fixing bugs, or enhancing documentation, your help is appreciated.

Please feel free to:
*   Report bugs or suggest features by opening an issue.
*   Submit pull requests with your contributions.

(Further details on coding standards, development setup, and testing procedures specific to contributions can be added here or in a separate `CONTRIBUTING.md` file.)

## License

`tensorus-models` is licensed under the MIT License. You can find the full license text in the `LICENSE` file in the root of this repository. (It's recommended to add a `LICENSE` file with the MIT license text if one doesn't already exist).

## Acknowledgements

(Optional: This section can be used to acknowledge any specific inspirations, datasets, or libraries that were crucial in the development of `tensorus-models`, beyond standard dependencies.)

## Citation

If you use `tensorus-models` in your research or work, we would appreciate a citation. Please use the following placeholder (or update with specific publication details if available):

```
@software{tensorus_models_2023,
  author = {{Tensorus Team and Contributors}},
  title = {tensorus-models: A Collection of Machine Learning Models for the Tensorus Ecosystem},
  year = {2023}, # Or the year of release/version used
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tensorus/tensorus-models}} # Replace with actual URL
}
```
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
