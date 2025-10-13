"""
Prediction engine for EpiScan ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
import os
import json
from .feature_extractor import FeatureExtractor
from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

class OutbreakPredictor:
    """Main prediction engine for outbreak detection"""
    
    def __init__(self, model_dir: str = "ml_models/saved_models"):
        self.model_dir = model_dir
        self.feature_extractor = FeatureExtractor()
        self.model_trainer = ModelTrainer(model_dir)
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_metadata = {}
        self.is_loaded = False
    
    def load_models(self, timestamp: str = None):
        """Load trained models and scalers"""
        try:
            self.model_trainer.load_models(timestamp)
            self.models = self.model_trainer.models
            self.scalers = self.model_trainer.scalers
            self.feature_names = self.model_trainer.feature_names
            self.model_metadata = self.model_trainer.model_versions
            self.is_loaded = True
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    def prepare_prediction_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for prediction"""
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Extract features
        features_df = self.feature_extractor.extract_all_features(data)
        
        # Select feature columns (same as training)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['original_id', 'county', 'date', 'outbreak_occurred']]
        
        # Ensure we have the same features as training
        missing_features = set(self.feature_names) - set(feature_columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with zero values
            for feature in missing_features:
                features_df[feature] = 0
        
        # Select and order features
        X = features_df[self.feature_names].values
        
        # Scale features
        X_scaled = self.scalers['main'].transform(X)
        
        return X_scaled, features_df
    
    def predict_outbreak_probability(self, data: pd.DataFrame, 
                                   model_name: str = 'random_forest') -> Dict[str, Any]:
        """Predict outbreak probability for given data"""
        if not self.is_loaded:
            self.load_models()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        # Prepare data
        X_scaled, features_df = self.prepare_prediction_data(data)
        
        # Get model
        model = self.models[model_name]
        
        # Make predictions
        if model_name == 'tensorflow':
            probabilities = model.predict(X_scaled, verbose=0)
            if probabilities.shape[1] == 1:
                probabilities = probabilities.flatten()
            else:
                probabilities = probabilities[:, 1]
        else:
            probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Create results
        results = []
        for i, (idx, row) in enumerate(features_df.iterrows()):
            result = {
                'original_id': row.get('original_id', i),
                'county': row.get('county', ''),
                'date': row.get('date', datetime.now()),
                'outbreak_probability': float(probabilities[i]),
                'risk_level': self._determine_risk_level(probabilities[i]),
                'model_used': model_name,
                'prediction_timestamp': datetime.now().isoformat()
            }
            results.append(result)
        
        return {
            'predictions': results,
            'model_name': model_name,
            'total_predictions': len(results),
            'average_probability': float(np.mean(probabilities)),
            'high_risk_count': sum(1 for p in probabilities if p > 0.7)
        }
    
    def predict_county_outbreak(self, county_data: Dict[str, Any], 
                              model_name: str = 'random_forest') -> Dict[str, Any]:
        """Predict outbreak probability for a specific county"""
        # Convert county data to DataFrame
        data_df = pd.DataFrame([county_data])
        
        # Get prediction
        prediction_result = self.predict_outbreak_probability(data_df, model_name)
        
        if prediction_result['predictions']:
            prediction = prediction_result['predictions'][0]
            
            # Add county-specific analysis
            county_analysis = {
                'county': prediction['county'],
                'outbreak_probability': prediction['outbreak_probability'],
                'risk_level': prediction['risk_level'],
                'confidence': self._calculate_confidence(prediction['outbreak_probability']),
                'recommendations': self._generate_recommendations(prediction['outbreak_probability'], prediction['risk_level']),
                'model_used': model_name,
                'prediction_date': prediction['prediction_timestamp']
            }
            
            return county_analysis
        
        return {'error': 'No prediction generated'}
    
    def batch_predict(self, data: pd.DataFrame, 
                     model_name: str = 'random_forest') -> pd.DataFrame:
        """Perform batch predictions on a dataset"""
        prediction_result = self.predict_outbreak_probability(data, model_name)
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(prediction_result['predictions'])
        
        return predictions_df
    
    def ensemble_predict(self, data: pd.DataFrame, 
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Make ensemble predictions using all available models"""
        if not self.is_loaded:
            self.load_models()
        
        if weights is None:
            # Equal weights for all models
            weights = {name: 1.0 for name in self.models.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {name: weight/total_weight for name, weight in weights.items()}
        
        # Get predictions from all models
        all_predictions = {}
        for model_name in self.models.keys():
            try:
                pred_result = self.predict_outbreak_probability(data, model_name)
                all_predictions[model_name] = [p['outbreak_probability'] for p in pred_result['predictions']]
            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {str(e)}")
                continue
        
        if not all_predictions:
            raise ValueError("No models available for ensemble prediction")
        
        # Calculate weighted average
        ensemble_probs = np.zeros(len(data))
        for model_name, probs in all_predictions.items():
            ensemble_probs += np.array(probs) * weights[model_name]
        
        # Create ensemble results
        results = []
        for i, (idx, row) in enumerate(data.iterrows()):
            result = {
                'original_id': i,
                'county': row.get('county', ''),
                'date': row.get('created_at', row.get('published_at', datetime.now())),
                'outbreak_probability': float(ensemble_probs[i]),
                'risk_level': self._determine_risk_level(ensemble_probs[i]),
                'model_used': 'ensemble',
                'individual_predictions': {name: float(probs[i]) for name, probs in all_predictions.items()},
                'prediction_timestamp': datetime.now().isoformat()
            }
            results.append(result)
        
        return {
            'predictions': results,
            'model_name': 'ensemble',
            'model_weights': weights,
            'total_predictions': len(results),
            'average_probability': float(np.mean(ensemble_probs)),
            'high_risk_count': sum(1 for p in ensemble_probs if p > 0.7)
        }
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability"""
        if probability >= 0.8:
            return 'critical'
        elif probability >= 0.6:
            return 'high'
        elif probability >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence(self, probability: float) -> float:
        """Calculate confidence score based on probability"""
        # Higher confidence for extreme probabilities (close to 0 or 1)
        distance_from_0_5 = abs(probability - 0.5)
        confidence = 2 * distance_from_0_5  # Scale to 0-1
        return min(confidence, 1.0)
    
    def _generate_recommendations(self, probability: float, risk_level: str) -> List[str]:
        """Generate recommendations based on risk level"""
        recommendations = []
        
        if risk_level == 'critical':
            recommendations.extend([
                "Immediate public health intervention required",
                "Alert all healthcare facilities in the area",
                "Consider emergency response protocols",
                "Increase surveillance and monitoring"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "Increase monitoring and surveillance",
                "Prepare healthcare resources",
                "Consider targeted public health campaigns",
                "Coordinate with local health authorities"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "Continue regular monitoring",
                "Prepare contingency plans",
                "Monitor for trend changes",
                "Maintain awareness"
            ])
        else:
            recommendations.extend([
                "Continue routine surveillance",
                "Monitor for any changes",
                "Maintain preparedness"
            ])
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        if not self.is_loaded:
            return {'error': 'No models loaded'}
        
        return {
            'loaded_models': list(self.models.keys()),
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_metadata': self.model_metadata,
            'is_loaded': self.is_loaded
        }
    
    def predict_with_confidence_interval(self, data: pd.DataFrame, 
                                       confidence_level: float = 0.95) -> Dict[str, Any]:
        """Make predictions with confidence intervals using ensemble"""
        # Get ensemble predictions
        ensemble_result = self.ensemble_predict(data)
        
        # Calculate confidence intervals (simplified approach)
        probabilities = [p['outbreak_probability'] for p in ensemble_result['predictions']]
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        
        # Calculate confidence interval
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin_of_error = z_score * std_prob / np.sqrt(len(probabilities))
        
        confidence_interval = {
            'lower_bound': max(0, mean_prob - margin_of_error),
            'upper_bound': min(1, mean_prob + margin_of_error),
            'mean': mean_prob,
            'std': std_prob,
            'confidence_level': confidence_level
        }
        
        # Add confidence interval to each prediction
        for prediction in ensemble_result['predictions']:
            prediction['confidence_interval'] = confidence_interval
        
        ensemble_result['confidence_interval'] = confidence_interval
        
        return ensemble_result


