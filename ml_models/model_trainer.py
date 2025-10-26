"""
Model training module for EpiScan ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import joblib
import json
import os
import logging

# Try to import TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. TensorFlow models will be skipped.")
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate ML models for outbreak prediction"""
    
    def __init__(self, model_dir: str = "ml_models/saved_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Model configurations
        self.rf_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.nn_config = {
            'hidden_layer_sizes': (100, 50, 25),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'max_iter': 1000,
            'random_state': 42
        }
        
        self.tf_config = {
            'input_dim': None,  # Will be set based on feature count
            'layers': [128, 64, 32, 16],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'patience': 10
        }
        
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_versions = {}
    
    def create_tensorflow_model(self, input_dim: int, num_classes: int = 2):
        """Create a TensorFlow neural network model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.tf_config['layers'][0], activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(self.tf_config['dropout_rate']))
        
        # Hidden layers
        for units in self.tf_config['layers'][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.tf_config['dropout_rate']))
        
        # Output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.tf_config['learning_rate']),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Compute class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Create and train model
        rf_model = RandomForestClassifier(**self.rf_config, class_weight=class_weight_dict)
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = rf_model.score(X_train, y_train)
        val_score = rf_model.score(X_val, y_val)
        
        # Predictions
        y_train_pred = rf_model.predict(X_train)
        y_val_pred = rf_model.predict(X_val)
        y_val_proba = rf_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        val_auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.0
        
        metrics = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'val_auc': val_auc,
            'feature_importance': rf_model.feature_importances_.tolist()
        }
        
        self.models['random_forest'] = rf_model
        logger.info(f"Random Forest - Train: {train_score:.4f}, Val: {val_score:.4f}, AUC: {val_auc:.4f}")
        
        return metrics
    
    def train_neural_network_sklearn(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train scikit-learn Neural Network model"""
        logger.info("Training scikit-learn Neural Network model...")
        
        # Create and train model
        nn_model = MLPClassifier(**self.nn_config)
        nn_model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = nn_model.score(X_train, y_train)
        val_score = nn_model.score(X_val, y_val)
        
        # Predictions
        y_train_pred = nn_model.predict(X_train)
        y_val_pred = nn_model.predict(X_val)
        y_val_proba = nn_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        val_auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.0
        
        metrics = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'val_auc': val_auc,
            'n_iterations': nn_model.n_iter_,
            'loss_curve': nn_model.loss_curve_.tolist() if hasattr(nn_model, 'loss_curve_') and hasattr(nn_model.loss_curve_, 'tolist') else []
        }
        
        self.models['neural_network'] = nn_model
        logger.info(f"Neural Network - Train: {train_score:.4f}, Val: {val_score:.4f}, AUC: {val_auc:.4f}")
        
        return metrics
    
    def train_tensorflow_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train TensorFlow Neural Network model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping TensorFlow model.")
            return {'error': 'TensorFlow not installed'}
        
        logger.info("Training TensorFlow Neural Network model...")
        
        # Create model
        input_dim = X_train.shape[1]
        tf_model = self.create_tensorflow_model(input_dim)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.tf_config['patience'],
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = tf_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.tf_config['epochs'],
            batch_size=self.tf_config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_loss, train_acc = tf_model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = tf_model.evaluate(X_val, y_val, verbose=0)
        
        # Predictions
        y_val_proba = tf_model.predict(X_val, verbose=0)
        if y_val_proba.shape[1] == 1:  # Binary classification
            y_val_proba = y_val_proba.flatten()
        else:
            y_val_proba = y_val_proba[:, 1]  # Take positive class probability
        
        val_auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.0
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_auc': val_auc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epochs_trained': len(history.history['loss']),
            'history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            }
        }
        
        self.models['tensorflow'] = tf_model
        logger.info(f"TensorFlow - Train: {train_acc:.4f}, Val: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        return metrics
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            model_type: str = 'random_forest') -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearchCV"""
        logger.info(f"Performing hyperparameter tuning for {model_type}...")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        elif model_type == 'neural_network':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
            model = MLPClassifier(random_state=42, max_iter=1000)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Train all models and return comprehensive results"""
        logger.info("Starting training of all models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Train models
        results = {}
        
        # Random Forest
        try:
            rf_metrics = self.train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
            results['random_forest'] = rf_metrics
        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}")
            results['random_forest'] = {'error': str(e)}
        
        # Neural Network (scikit-learn)
        try:
            nn_metrics = self.train_neural_network_sklearn(X_train_scaled, y_train, X_val_scaled, y_val)
            results['neural_network'] = nn_metrics
        except Exception as e:
            logger.error(f"Neural Network training failed: {str(e)}")
            results['neural_network'] = {'error': str(e)}
        
        # TensorFlow Neural Network
        try:
            tf_metrics = self.train_tensorflow_model(X_train_scaled, y_train, X_val_scaled, y_val)
            results['tensorflow'] = tf_metrics
        except Exception as e:
            logger.error(f"TensorFlow training failed: {str(e)}")
            results['tensorflow'] = {'error': str(e)}
        
        # Final evaluation on test set
        test_results = self.evaluate_on_test_set(X_test_scaled, y_test)
        results['test_evaluation'] = test_results
        
        # Save models
        self.save_models()
        
        return results
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate all trained models on test set"""
        logger.info("Evaluating models on test set...")
        
        test_results = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'tensorflow':
                    y_pred_proba = model.predict(X_test, verbose=0)
                    if y_pred_proba.shape[1] == 1:
                        y_pred_proba = y_pred_proba.flatten()
                    else:
                        y_pred_proba = y_pred_proba[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = (y_pred == y_test).mean()
                auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
                
                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                test_results[model_name] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1_score': report['weighted avg']['f1-score'],
                    'confusion_matrix': cm.tolist(),
                    'classification_report': report
                }
                
                logger.info(f"{model_name} - Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
                
            except Exception as e:
                logger.error(f"Test evaluation failed for {model_name}: {str(e)}")
                test_results[model_name] = {'error': str(e)}
        
        return test_results
    
    def print_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Print confusion matrix in a readable format"""
        print(f"\n{model_name} Confusion Matrix:")
        print("=" * 40)
        print(f"{'':>10} {'Predicted':>20}")
        print(f"{'':>10} {'No Outbreak':>12} {'Outbreak':>8}")
        print(f"{'Actual':>10} {'No Outbreak':>12} {cm[0,0]:>8} {cm[0,1]:>8}")
        print(f"{'':>10} {'Outbreak':>12} {cm[1,0]:>8} {cm[1,1]:>8}")
        print("=" * 40)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"True Negatives (TN):  {tn:>8} - Correctly predicted no outbreak")
        print(f"False Positives (FP): {fp:>8} - Incorrectly predicted outbreak")
        print(f"False Negatives (FN): {fn:>8} - Missed actual outbreak")
        print(f"True Positives (TP):  {tp:>8} - Correctly predicted outbreak")
        print(f"Sensitivity (Recall): {sensitivity:.3f} - How well we catch outbreaks")
        print(f"Specificity:         {specificity:.3f} - How well we avoid false alarms")
        print()
    
    def save_models(self):
        """Save trained models and scalers"""
        logger.info("Saving models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            if model_name == 'tensorflow':
                model_path = os.path.join(self.model_dir, f"{model_name}_{timestamp}.h5")
                model.save(model_path)
            else:
                model_path = os.path.join(self.model_dir, f"{model_name}_{timestamp}.joblib")
                joblib.dump(model, model_path)
            
            self.model_versions[model_name] = {
                'path': model_path,
                'timestamp': timestamp,
                'version': timestamp
            }
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, f"scaler_{timestamp}.joblib")
        joblib.dump(self.scalers['main'], scaler_path)
        
        # Save metadata
        metadata = {
            'model_versions': self.model_versions,
            'feature_names': self.feature_names,
            'training_timestamp': timestamp,
            'model_configs': {
                'random_forest': self.rf_config,
                'neural_network': self.nn_config,
                'tensorflow': self.tf_config
            }
        }
        
        metadata_path = os.path.join(self.model_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self, timestamp: str = None):
        """Load trained models"""
        if timestamp is None:
            # Find latest timestamp
            metadata_files = [f for f in os.listdir(self.model_dir) if f.startswith('metadata_')]
            if not metadata_files:
                raise FileNotFoundError("No trained models found")
            timestamp = sorted(metadata_files)[-1].replace('metadata_', '').replace('.json', '')
        
        metadata_path = os.path.join(self.model_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.model_versions = metadata['model_versions']
        self.feature_names = metadata['feature_names']
        
        # Load models
        for model_name, model_info in self.model_versions.items():
            model_path = model_info['path']
            if model_name == 'tensorflow':
                self.models[model_name] = tf.keras.models.load_model(model_path)
            else:
                self.models[model_name] = joblib.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(self.model_dir, f"scaler_{timestamp}.joblib")
        self.scalers['main'] = joblib.load(scaler_path)
        
        logger.info(f"Models loaded from {timestamp}")
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance"""
        summary = {
            'available_models': list(self.models.keys()),
            'model_versions': self.model_versions,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        return summary
