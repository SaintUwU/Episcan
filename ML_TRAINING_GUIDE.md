# EpiScan ML Model Training Guide

This guide explains how to train the machine learning models for disease outbreak prediction in the EpiScan system.

## 🎯 Overview

The EpiScan system uses multiple machine learning models to predict disease outbreaks:

- **Random Forest**: Ensemble method for robust predictions
- **Neural Network (scikit-learn)**: Multi-layer perceptron for pattern recognition
- **TensorFlow Neural Network**: Deep learning model for complex patterns

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have:
- Python 3.8+ installed
- All dependencies installed: `pip install -r requirements.txt`
- Database initialized: `python app.py init-db`
- Some data collected: `python app.py collect-data`

### 2. Basic Training

```bash
# Train all models with default settings
python app.py train-models

# Train with custom parameters
python ml_models/train_models.py --days 120 --hyperparameter-tuning
```

### 3. Make Predictions

```bash
# Make predictions using trained models
python app.py make-predictions

# Evaluate model performance
python app.py evaluate-models
```

## 📊 Training Process

### Data Collection
The system automatically collects training data from:
- **Twitter/X**: Social media posts about health topics
- **News Articles**: Kenyan news sources
- **Google Trends**: Search interest data
- **WHO Reports**: Official health reports

### Feature Extraction
The system extracts 20+ features including:
- **Text Features**: Length, word count, sentiment scores
- **Temporal Features**: Day of week, time of day, recency
- **Geographical Features**: County encoding, urban/rural classification
- **Social Features**: Engagement metrics, retweet counts
- **Health Features**: Disease keyword counts, health relevance scores

### Model Training
1. **Data Splitting**: 80% training, 20% testing
2. **Feature Scaling**: StandardScaler for neural networks
3. **Class Balancing**: Handles imbalanced outbreak data
4. **Cross-Validation**: 5-fold CV for robust evaluation
5. **Hyperparameter Tuning**: GridSearchCV for optimal parameters

## 🔧 Advanced Usage

### Custom Training Parameters

```bash
# Train specific models only
python ml_models/train_models.py --models random_forest tensorflow

# Use more data for training
python ml_models/train_models.py --days 180

# Skip hyperparameter tuning (faster)
python ml_models/train_models.py --no-hyperparameter-tuning

# Don't save training data
python ml_models/train_models.py --no-save-data
```

### Training with Custom Data

```python
from ml_models.data_processor import DataProcessor
from ml_models.model_trainer import ModelTrainer

# Prepare custom dataset
processor = DataProcessor()
df, features = processor.prepare_ml_dataset(days_back=90)

# Train models
trainer = ModelTrainer()
X, y = processor.feature_extractor.prepare_training_data(df)
results = trainer.train_all_models(X, y)
```

### Making Predictions Programmatically

```python
from ml_models.predictor import OutbreakPredictor
import pandas as pd

# Initialize predictor
predictor = OutbreakPredictor()
predictor.load_models()

# Prepare data
data = pd.DataFrame([{
    'text': 'Many people reporting flu symptoms in Nairobi',
    'county': 'nairobi',
    'created_at': '2024-01-15',
    'source_type': 'twitter'
}])

# Make predictions
predictions = predictor.batch_predict(data)
print(predictions)
```

## 📈 Model Performance

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

### Expected Performance
- **Random Forest**: 85-90% accuracy, good interpretability
- **Neural Network**: 80-88% accuracy, good for complex patterns
- **TensorFlow**: 82-90% accuracy, best for large datasets

## 🗂️ File Structure

```
ml_models/
├── __init__.py              # Package initialization
├── feature_extractor.py     # Feature extraction logic
├── model_trainer.py         # Model training and evaluation
├── predictor.py             # Prediction engine
├── data_processor.py        # Data preprocessing pipeline
├── train_models.py          # Training CLI script
└── saved_models/            # Trained model files
    ├── random_forest_*.joblib
    ├── neural_network_*.joblib
    ├── tensorflow_*.h5
    ├── scaler_*.joblib
    └── metadata_*.json
```

## 🔍 Monitoring and Debugging

### Training Logs
Training progress is logged with timestamps:
```
2024-01-15 10:30:15 - INFO - Starting model training process...
2024-01-15 10:30:20 - INFO - Collected 1500 records for training
2024-01-15 10:30:25 - INFO - Extracted 20 features
2024-01-15 10:30:30 - INFO - Training Random Forest model...
```

### Common Issues

#### 1. No Data Available
```
Error: No data available for training
```
**Solution**: Run data collection first:
```bash
python app.py collect-data
```

#### 2. Insufficient Class Diversity
```
Error: Insufficient class diversity for training
```
**Solution**: Collect more data or adjust target creation logic

#### 3. Memory Issues
```
Error: Out of memory during training
```
**Solution**: Reduce dataset size or use smaller models:
```bash
python ml_models/train_models.py --days 30
```

#### 4. Model Loading Errors
```
Error: Failed to load models
```
**Solution**: Check if models exist and retrain:
```bash
ls ml_models/saved_models/
python app.py train-models
```

## 🎛️ Configuration

### Model Parameters

#### Random Forest
```python
rf_config = {
    'n_estimators': 100,        # Number of trees
    'max_depth': 10,            # Maximum tree depth
    'min_samples_split': 5,     # Minimum samples to split
    'min_samples_leaf': 2,      # Minimum samples per leaf
    'random_state': 42          # For reproducibility
}
```

#### Neural Network (scikit-learn)
```python
nn_config = {
    'hidden_layer_sizes': (100, 50, 25),  # Hidden layer sizes
    'activation': 'relu',                 # Activation function
    'solver': 'adam',                     # Optimizer
    'alpha': 0.001,                       # L2 regularization
    'learning_rate': 'adaptive',          # Learning rate strategy
    'max_iter': 1000                      # Maximum iterations
}
```

#### TensorFlow Neural Network
```python
tf_config = {
    'layers': [128, 64, 32, 16],  # Hidden layer sizes
    'dropout_rate': 0.3,          # Dropout for regularization
    'learning_rate': 0.001,       # Learning rate
    'epochs': 100,                # Training epochs
    'batch_size': 32,             # Batch size
    'patience': 10                # Early stopping patience
}
```

## 🔄 Automated Training

### Scheduled Training
Set up automated model retraining:

```bash
# Add to crontab for daily retraining
0 2 * * * cd /path/to/episcan && python app.py train-models

# Weekly evaluation
0 3 * * 0 cd /path/to/episcan && python app.py evaluate-models
```

### CI/CD Integration
```yaml
# GitHub Actions example
name: Train Models
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train models
        run: python app.py train-models
```

## 📊 Model Comparison

| Model | Accuracy | Speed | Interpretability | Memory Usage |
|-------|----------|-------|------------------|--------------|
| Random Forest | 85-90% | Fast | High | Low |
| Neural Network | 80-88% | Medium | Medium | Medium |
| TensorFlow | 82-90% | Slow | Low | High |

## 🚨 Production Considerations

### Model Versioning
- Models are automatically versioned with timestamps
- Keep multiple versions for rollback capability
- Monitor model performance over time

### Performance Monitoring
```python
# Check model performance
from app.models.prediction_models import ModelPerformance

# Get latest performance metrics
latest_performance = ModelPerformance.query.order_by(
    ModelPerformance.evaluation_date.desc()
).first()
```

### A/B Testing
```python
# Compare model performance
from ml_models.predictor import OutbreakPredictor

predictor = OutbreakPredictor()
predictor.load_models()

# Use ensemble prediction for better accuracy
ensemble_result = predictor.ensemble_predict(data)
```

## 🎯 Best Practices

1. **Regular Retraining**: Retrain models weekly with fresh data
2. **Data Quality**: Ensure clean, relevant data for training
3. **Feature Engineering**: Continuously improve feature extraction
4. **Model Monitoring**: Track performance metrics over time
5. **Validation**: Use cross-validation for robust evaluation
6. **Documentation**: Keep track of model versions and changes

## 📞 Support

For issues or questions:
1. Check the training logs for error messages
2. Verify data availability and quality
3. Ensure all dependencies are installed
4. Check database connectivity
5. Review model configuration parameters

---

**EpiScan ML Training** - Building Intelligent Disease Surveillance Models


