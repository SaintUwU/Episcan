#!/usr/bin/env python3
"""
Training script for EpiScan ML models
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.data_processor import DataProcessor
from ml_models.feature_extractor import FeatureExtractor
from ml_models.model_trainer import ModelTrainer
from ml_models.predictor import OutbreakPredictor
from app import create_app, db
from app.models.prediction_models import ModelPerformance

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_models(days_back: int = 90, save_data: bool = True, 
                hyperparameter_tuning: bool = False, model_types: list = None):
    """Train all ML models"""
    logger.info("Starting model training process...")
    
    if model_types is None:
        model_types = ['random_forest', 'neural_network', 'tensorflow']
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        feature_extractor = FeatureExtractor()
        model_trainer = ModelTrainer()
        
        # Prepare data
        logger.info("Preparing training data...")
        df, feature_columns = data_processor.prepare_ml_dataset(days_back)
        
        if df.empty:
            logger.error("No data available for training")
            return False
        
        # Save training data if requested
        if save_data:
            data_file = data_processor.save_training_data(df)
            logger.info(f"Training data saved to {data_file}")
        
        # Extract features
        logger.info("Extracting features...")
        features_df = feature_extractor.extract_all_features(df)
        
        # Prepare training data
        X, y = feature_extractor.prepare_training_data(features_df)
        
        if len(np.unique(y)) < 2:
            logger.error("Insufficient class diversity for training")
            return False
        
        logger.info(f"Training data shape: {X.shape}, Target distribution: {np.bincount(y)}")
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            for model_type in ['random_forest', 'neural_network']:
                try:
                    tuning_results = model_trainer.hyperparameter_tuning(X, y, model_type)
                    logger.info(f"Best parameters for {model_type}: {tuning_results['best_params']}")
                except Exception as e:
                    logger.error(f"Hyperparameter tuning failed for {model_type}: {str(e)}")
        
        # Train models
        logger.info("Training models...")
        training_results = model_trainer.train_all_models(X, y)
        
        # Save model performance to database
        save_model_performance(training_results, model_trainer.feature_names)
        
        # Print results
        print_training_results(training_results)
        
        logger.info("Model training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False

def save_model_performance(results: dict, feature_names: list):
    """Save model performance metrics to database"""
    app = create_app()
    
    with app.app_context():
        try:
            for model_name, metrics in results.items():
                if model_name == 'test_evaluation' or 'error' in metrics:
                    continue
                
                # Get test evaluation if available
                test_metrics = results.get('test_evaluation', {}).get(model_name, {})
                
                performance = ModelPerformance(
                    model_name=model_name,
                    model_version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    evaluation_date=datetime.now().date(),
                    accuracy=test_metrics.get('accuracy', metrics.get('val_accuracy', 0.0)),
                    precision=test_metrics.get('precision', 0.0),
                    recall=test_metrics.get('recall', 0.0),
                    f1_score=test_metrics.get('f1_score', 0.0),
                    auc_roc=test_metrics.get('auc', metrics.get('val_auc', 0.0)),
                    confusion_matrix={},  # Could be added if needed
                    feature_importance=metrics.get('feature_importance', []),
                    evaluation_notes=f"Training completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                db.session.add(performance)
            
            db.session.commit()
            logger.info("Model performance metrics saved to database")
            
        except Exception as e:
            logger.error(f"Failed to save model performance: {str(e)}")
            db.session.rollback()

def print_training_results(results: dict):
    """Print training results in a formatted way"""
    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS")
    print("="*60)
    
    for model_name, metrics in results.items():
        if model_name == 'test_evaluation' or 'error' in metrics:
            continue
        
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        if 'error' in metrics:
            print(f"  Error: {metrics['error']}")
        else:
            print(f"  Training Accuracy: {metrics.get('train_accuracy', 0):.4f}")
            print(f"  Validation Accuracy: {metrics.get('val_accuracy', 0):.4f}")
            print(f"  Validation AUC: {metrics.get('val_auc', 0):.4f}")
            
            if 'epochs_trained' in metrics:
                print(f"  Epochs Trained: {metrics['epochs_trained']}")
            
            if 'n_iterations' in metrics:
                print(f"  Iterations: {metrics['n_iterations']}")
    
    # Test evaluation
    if 'test_evaluation' in results:
        print(f"\nTEST SET EVALUATION:")
        print("-" * 40)
        
        for model_name, test_metrics in results['test_evaluation'].items():
            if 'error' in test_metrics:
                print(f"  {model_name}: Error - {test_metrics['error']}")
            else:
                print(f"  {model_name}:")
                print(f"    Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                print(f"    AUC: {test_metrics.get('auc', 0):.4f}")
                print(f"    Precision: {test_metrics.get('precision', 0):.4f}")
                print(f"    Recall: {test_metrics.get('recall', 0):.4f}")
                print(f"    F1-Score: {test_metrics.get('f1_score', 0):.4f}")
    
    print("\n" + "="*60)

def evaluate_models(days_back: int = 30):
    """Evaluate existing models on recent data"""
    logger.info("Evaluating existing models...")
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        feature_extractor = FeatureExtractor()
        predictor = OutbreakPredictor()
        
        # Load models
        predictor.load_models()
        
        # Prepare evaluation data
        df, _ = data_processor.prepare_ml_dataset(days_back)
        
        if df.empty:
            logger.error("No data available for evaluation")
            return False
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = predictor.batch_predict(df)
        
        # Calculate evaluation metrics
        if 'outbreak_occurred' in df.columns:
            actual = df['outbreak_occurred'].values
            predicted = predictions['outbreak_probability'].values
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            threshold = 0.5
            predicted_binary = (predicted > threshold).astype(int)
            
            accuracy = accuracy_score(actual, predicted_binary)
            precision = precision_score(actual, predicted_binary, zero_division=0)
            recall = recall_score(actual, predicted_binary, zero_division=0)
            f1 = f1_score(actual, predicted_binary, zero_division=0)
            auc = roc_auc_score(actual, predicted) if len(np.unique(actual)) > 1 else 0.0
            
            print(f"\nEVALUATION RESULTS:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
        
        logger.info("Model evaluation completed")
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return False

def main():
    """Main function for CLI"""
    parser = argparse.ArgumentParser(description='Train EpiScan ML models')
    parser.add_argument('--days', type=int, default=90, 
                       help='Number of days of data to use for training')
    parser.add_argument('--no-save-data', action='store_true',
                       help='Do not save training data to file')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing models')
    parser.add_argument('--models', nargs='+', 
                       choices=['random_forest', 'neural_network', 'tensorflow'],
                       help='Specific models to train')
    
    args = parser.parse_args()
    
    if args.evaluate_only:
        success = evaluate_models(args.days)
    else:
        success = train_models(
            days_back=args.days,
            save_data=not args.no_save_data,
            hyperparameter_tuning=args.hyperparameter_tuning,
            model_types=args.models
        )
    
    if success:
        print("\n✅ Process completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Process failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()


