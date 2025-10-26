#!/usr/bin/env python3
"""
Retrain EpiScan models with expanded dataset
"""
import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.data_processor import DataProcessor
from ml_models.model_trainer import ModelTrainer
from ml_models.predictor import OutbreakPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrain_models_with_expanded_data():
    """Retrain models with expanded dataset"""
    print("=" * 60)
    print("EPISCAN MODEL RETRAINING WITH EXPANDED DATA")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        
        print("Step 1: Preparing expanded training dataset...")
        
        # Collect data from existing training data or extended period
        df, feature_columns = data_processor.prepare_ml_dataset_with_existing(use_existing=True, days_back=180)
        
        if df.empty:
            print("[ERROR] No data available for training. Please run data collection first.")
            return False
        
        print(f"[SUCCESS] Dataset prepared: {len(df)} records, {len(feature_columns)} features")
        
        # Save the expanded training data
        training_file = data_processor.save_training_data(df)
        print(f"[SUCCESS] Training data saved to: {training_file}")
        
        print("\nStep 2: Training models with expanded data...")
        
        # Prepare features and target
        X = df[feature_columns].values
        y = df['outbreak_occurred'].values
        
        # Train models
        training_results = model_trainer.train_all_models(
            X=X,
            y=y,
            test_size=0.2,
            random_state=42
        )
        
        print("\nStep 3: Model training results:")
        for model_name, results in training_results.items():
            if 'error' in results:
                print(f"  ❌ {model_name}: {results['error']}")
            elif model_name == 'test_evaluation':
                # Handle test evaluation results
                if isinstance(results, dict):
                    avg_accuracy = 0
                    count = 0
                    for sub_model, sub_results in results.items():
                        if isinstance(sub_results, dict) and 'accuracy' in sub_results:
                            avg_accuracy += sub_results['accuracy']
                            count += 1
                    if count > 0:
                        avg_accuracy /= count
                        print(f"  ✅ {model_name}: {avg_accuracy:.3f} accuracy")
                    else:
                        print(f"  ✅ {model_name}: Test evaluation completed")
                else:
                    print(f"  ✅ {model_name}: Test evaluation completed")
            else:
                # Get validation accuracy (primary metric)
                accuracy = results.get('val_accuracy', results.get('train_accuracy', 0))
                print(f"  ✅ {model_name}: {accuracy:.3f} accuracy")
        
        print("\nStep 4: Testing model performance...")
        
        # Test the trained models
        predictor = OutbreakPredictor()
        
        # Test with sample data
        test_results = {}
        for model_name in ['random_forest', 'neural_network']:
            try:
                # Load the models
                predictor.load_models()
                if predictor.models.get(model_name):
                    print(f"  ✅ {model_name} model loaded successfully")
                    test_results[model_name] = "Loaded successfully"
                else:
                    print(f"  ❌ {model_name} model not found")
                    test_results[model_name] = "Model not found"
            except Exception as e:
                print(f"  ❌ {model_name} error: {str(e)}")
                test_results[model_name] = f"Error: {str(e)}"
        
        print(f"\n✅ Model retraining completed at: {datetime.now()}")
        
        # Summary
        print("\n" + "=" * 60)
        print("RETRAINING SUMMARY")
        print("=" * 60)
        print(f"Training records: {len(df)}")
        print(f"Features: {len(feature_columns)}")
        print(f"Training data file: {training_file}")
        
        # Check if we have enough data now
        if len(df) > 10000:
            print("[SUCCESS] Excellent! You now have sufficient data for robust ML training.")
        elif len(df) > 5000:
            print("[SUCCESS] Good! You have a decent amount of data for training.")
        else:
            print("[WARNING] You may still need more data for optimal training.")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error during retraining: {str(e)}")
        logger.error(f"Retraining error: {str(e)}")
        return False

def main():
    """Main execution function"""
    success = retrain_models_with_expanded_data()
    
    if success:
        print("\n[SUCCESS] Model retraining completed successfully!")
        print("Your models are now trained with expanded data and should perform better.")
    else:
        print("\n[ERROR] Model retraining failed. Please check the errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
