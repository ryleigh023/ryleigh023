#!/usr/bin/env python3
"""
Regional Avalanche Model Training System

Train region-specific avalanche risk models using local data collected from:
- Regional avalanche centers (CAIC, Avalanche Canada, etc.)
- Historical observations and bulletins
- Field observations
- Weather station data

Author: AI Python Developer
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Import our data collection system
from regional_data_collector import RegionalDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegionalModelTrainer:
    """Train and validate region-specific avalanche risk models"""
    
    def __init__(self, region: str, data_dir: str = "/workspace/regional_data"):
        self.region = region
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        }
        
        # Feature engineering configurations
        self.regional_features = {
            'colorado': ['elevation', 'aspect_numerical', 'slope_angle', 'new_snow_24h', 
                        'wind_speed', 'air_temperature', 'steep_slope', 'high_wind'],
            'british_columbia': ['elevation', 'aspect_numerical', 'slope_angle', 'new_snow_24h',
                               'wind_speed', 'air_temperature', 'precipitation_24h'],
            'utah': ['elevation', 'aspect_numerical', 'slope_angle', 'new_snow_24h',
                    'wind_speed', 'air_temperature', 'total_depth'],
            'washington': ['elevation', 'aspect_numerical', 'slope_angle', 'new_snow_24h',
                         'wind_speed', 'air_temperature', 'precipitation_24h']
        }
        
    def load_regional_data(self) -> pd.DataFrame:
        """Load collected regional data for training"""
        
        # Try to load from existing training dataset
        training_file = self.data_dir / f"{self.region.lower()}_training_data.csv"
        
        if training_file.exists():
            logger.info(f"Loading existing training data: {training_file}")
            df = pd.read_csv(training_file)
        else:
            logger.info("No existing training data found, collecting new data...")
            # Collect fresh data
            collector = RegionalDataCollector(self.region, str(self.data_dir))
            collector.collect_all_regional_data(days_back=365)  # Get more historical data
            df = collector.generate_training_dataset()
        
        logger.info(f"Loaded {len(df)} regional observations")
        return df
    
    def engineer_regional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create region-specific features for training"""
        
        logger.info("Engineering regional features...")
        
        # Base feature engineering
        df = df.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Create derived features
        df['elevation_normalized'] = df['elevation'] / 4000.0  # Normalize elevation
        df['wind_load'] = df['wind_speed'] * df['new_snow_24h'].fillna(0)
        df['temperature_gradient'] = df['air_temperature'] - (-5.0)  # Relative to typical freezing
        
        # Regional-specific features
        if self.region.lower() == 'colorado':
            # Colorado-specific features (high altitude, continental climate)
            df['high_altitude'] = (df['elevation'] > 3500).astype(int)
            df['continental_exposure'] = ((df['aspect_numerical'] > 0.4) & 
                                        (df['aspect_numerical'] < 0.8)).astype(int)  # E-S-W exposure
            df['wind_slab_risk'] = (df['wind_speed'] > 30) & (df['new_snow_24h'] > 15)
            
        elif self.region.lower() == 'british_columbia':
            # BC-specific features (maritime snowpack, varied terrain)
            df['maritime_conditions'] = (df['air_temperature'] > -10).astype(int)
            df['coastal_influence'] = (df['precipitation_24h'] > 10).astype(int)
            df['temperature_instability'] = (df['air_temperature'] > -2).astype(int)
            
        # Create stability indicators
        df['instability_score'] = (
            (df['new_snow_24h'] > 20).astype(int) * 2 +
            (df['wind_speed'] > 25).astype(int) * 2 +
            (df['air_temperature'] > -2).astype(int) * 1 +
            (df['recent_avalanche']).astype(int) * 3
        )
        
        # Select regional features
        available_features = self.regional_features.get(self.region.lower(), 
                                                       ['elevation', 'slope_angle', 'new_snow_24h'])
        
        # Ensure all features exist in dataframe
        feature_columns = []
        for feature in available_features:
            if feature in df.columns:
                feature_columns.append(feature)
            else:
                logger.warning(f"Feature {feature} not available in data")
        
        # Add derived features that are always useful
        always_include = ['instability_score', 'elevation_normalized', 'wind_load']
        for feature in always_include:
            if feature in df.columns and feature not in feature_columns:
                feature_columns.append(feature)
        
        logger.info(f"Using {len(feature_columns)} features for training: {feature_columns}")
        
        return df, feature_columns
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate the quality of training data"""
        
        quality_report = {
            'total_samples': len(df),
            'missing_risk_labels': df['risk_level'].isna().sum(),
            'class_distribution': df['risk_level'].value_counts().to_dict(),
            'missing_features': {},
            'temporal_coverage': {},
            'spatial_coverage': {}
        }
        
        # Check missing values in key features
        key_features = ['elevation', 'slope_angle', 'new_snow_24h', 'wind_speed']
        for feature in key_features:
            if feature in df.columns:
                missing_pct = (df[feature].isna().sum() / len(df)) * 100
                quality_report['missing_features'][feature] = f"{missing_pct:.1f}%"
        
        # Check temporal coverage
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            date_range = df['timestamp'].max() - df['timestamp'].min()
            quality_report['temporal_coverage'] = {
                'date_range_days': date_range.days,
                'earliest': df['timestamp'].min().strftime('%Y-%m-%d'),
                'latest': df['timestamp'].max().strftime('%Y-%m-%d')
            }
        
        # Check spatial coverage
        if 'location_name' in df.columns:
            unique_locations = df['location_name'].nunique()
            quality_report['spatial_coverage'] = {
                'unique_locations': unique_locations,
                'locations': df['location_name'].value_counts().head().to_dict()
            }
        
        # Data quality warnings
        warnings = []
        if quality_report['total_samples'] < 50:
            warnings.append(f"Low sample count: {quality_report['total_samples']} (recommend >100)")
        
        if quality_report['missing_risk_labels'] > 0:
            warnings.append(f"{quality_report['missing_risk_labels']} samples missing risk labels")
        
        # Check class balance
        class_counts = list(quality_report['class_distribution'].values())
        if len(class_counts) > 1:
            imbalance_ratio = max(class_counts) / min(class_counts)
            if imbalance_ratio > 5:
                warnings.append(f"Severe class imbalance: ratio {imbalance_ratio:.1f}")
        
        quality_report['warnings'] = warnings
        
        logger.info(f"Data quality validation completed: {len(warnings)} warnings")
        return quality_report
    
    def train_multiple_models(self, X: np.ndarray, y: np.ndarray, 
                            feature_names: List[str]) -> Dict:
        """Train and compare multiple model types"""
        
        logger.info("Training multiple model types...")
        
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Grid search for best parameters
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'],
                    cv=min(5, len(np.unique(y_train))),  # Adjust CV folds based on classes
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train_scaled, y_train)
                
                # Best model predictions
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_scaled)
                y_pred_proba = best_model.predict_proba(X_test_scaled) if hasattr(best_model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=3)
                
                results[model_name] = {
                    'model': best_model,
                    'scaler': scaler,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'best_params': grid_search.best_params_,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'test_indices': np.arange(len(y_test)),
                    'y_test': y_test,
                    'feature_names': feature_names
                }
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            results['best_model'] = best_model_name
            logger.info(f"Best model: {best_model_name} (accuracy: {results[best_model_name]['accuracy']:.3f})")
        
        return results
    
    def evaluate_regional_model(self, results: Dict, quality_report: Dict) -> None:
        """Create comprehensive evaluation of the regional model"""
        
        if not results or 'best_model' not in results:
            logger.error("No valid models to evaluate")
            return
        
        best_model_name = results['best_model']
        best_result = results[best_model_name]
        
        logger.info("Creating model evaluation report...")
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Regional Avalanche Model Evaluation - {self.region}', fontsize=16)
        
        # 1. Model comparison
        ax1 = axes[0, 0]
        model_names = [name for name in results.keys() if name != 'best_model']
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        ax1.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        ax1.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8)
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name.replace('_', ' ').title() for name in model_names], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confusion matrix
        ax2 = axes[0, 1]
        cm = confusion_matrix(best_result['y_test'], best_result['predictions'])
        class_names = ['Stable', 'Borderline', 'Unstable']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=class_names, yticklabels=class_names)
        ax2.set_title(f'Confusion Matrix - {best_model_name.replace("_", " ").title()}')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        # 3. Feature importance (if available)
        ax3 = axes[0, 2]
        if hasattr(best_result['model'], 'feature_importances_'):
            importances = best_result['model'].feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10
            ax3.barh(range(len(indices)), importances[indices])
            ax3.set_yticks(range(len(indices)))
            ax3.set_yticklabels([best_result['feature_names'][i] for i in indices])
            ax3.set_title('Top 10 Feature Importances')
            ax3.set_xlabel('Importance')
        else:
            ax3.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Importance')
        
        # 4. Class distribution
        ax4 = axes[1, 0]
        class_dist = quality_report['class_distribution']
        ax4.pie(class_dist.values(), labels=class_dist.keys(), autopct='%1.1f%%')
        ax4.set_title('Training Data Class Distribution')
        
        # 5. Data quality metrics
        ax5 = axes[1, 1]
        quality_metrics = [
            f"Total Samples: {quality_report['total_samples']}",
            f"Missing Labels: {quality_report['missing_risk_labels']}",
            f"Unique Locations: {quality_report['spatial_coverage'].get('unique_locations', 'N/A')}",
            f"Date Range: {quality_report['temporal_coverage'].get('date_range_days', 'N/A')} days",
            f"Best Model: {best_model_name.replace('_', ' ').title()}",
            f"Test Accuracy: {best_result['accuracy']:.3f}",
            f"CV Score: {best_result['cv_mean']:.3f}±{best_result['cv_std']:.3f}",
            f"F1 Score: {best_result['f1_score']:.3f}"
        ]
        
        ax5.axis('off')
        for i, metric in enumerate(quality_metrics):
            ax5.text(0.1, 0.9 - i*0.1, metric, transform=ax5.transAxes, fontsize=10)
        ax5.set_title('Model & Data Summary')
        
        # 6. Prediction confidence distribution
        ax6 = axes[1, 2]
        if best_result['probabilities'] is not None:
            max_probs = np.max(best_result['probabilities'], axis=1)
            ax6.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Maximum Prediction Probability')
            ax6.set_ylabel('Count')
            ax6.set_title('Prediction Confidence Distribution')
            ax6.axvline(np.mean(max_probs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(max_probs):.3f}')
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'Probability data\nnot available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Prediction Confidence')
        
        plt.tight_layout()
        
        # Save evaluation plot
        eval_plot_path = self.models_dir / f"{self.region.lower()}_model_evaluation.png"
        plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed classification report
        print("\n" + "="*60)
        print(f"REGIONAL MODEL EVALUATION - {self.region.upper()}")
        print("="*60)
        print(f"Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"Best Parameters: {best_result['best_params']}")
        print("\nClassification Report:")
        print(classification_report(best_result['y_test'], best_result['predictions'],
                                  target_names=['Stable', 'Borderline', 'Unstable']))
        
        # Data quality warnings
        if quality_report['warnings']:
            print("\n⚠️  DATA QUALITY WARNINGS:")
            for warning in quality_report['warnings']:
                print(f"  • {warning}")
        
        print(f"\n📊 Evaluation plot saved: {eval_plot_path}")
    
    def save_regional_model(self, results: Dict, quality_report: Dict) -> str:
        """Save the best regional model for deployment"""
        
        if not results or 'best_model' not in results:
            raise ValueError("No valid models to save")
        
        best_model_name = results['best_model']
        best_result = results[best_model_name]
        
        # Create model package
        model_package = {
            'model': best_result['model'],
            'scaler': best_result['scaler'],
            'feature_names': best_result['feature_names'],
            'model_type': best_model_name,
            'region': self.region,
            'training_date': datetime.now().isoformat(),
            'performance_metrics': {
                'accuracy': best_result['accuracy'],
                'precision': best_result['precision'],
                'recall': best_result['recall'],
                'f1_score': best_result['f1_score'],
                'cv_mean': best_result['cv_mean'],
                'cv_std': best_result['cv_std']
            },
            'training_data_quality': quality_report,
            'best_parameters': best_result['best_params']
        }
        
        # Save model
        model_path = self.models_dir / f"{self.region.lower()}_regional_model.pkl"
        joblib.dump(model_package, model_path)
        
        logger.info(f"Regional model saved: {model_path}")
        
        # Create model info file
        info_path = self.models_dir / f"{self.region.lower()}_model_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Regional Avalanche Model - {self.region}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model Type: {best_model_name.replace('_', ' ').title()}\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training Samples: {quality_report['total_samples']}\n")
            f.write(f"Test Accuracy: {best_result['accuracy']:.3f}\n")
            f.write(f"Cross-Validation: {best_result['cv_mean']:.3f}±{best_result['cv_std']:.3f}\n")
            f.write(f"Features Used: {len(best_result['feature_names'])}\n")
            f.write(f"Feature List: {', '.join(best_result['feature_names'])}\n\n")
            f.write("Best Parameters:\n")
            for param, value in best_result['best_params'].items():
                f.write(f"  {param}: {value}\n")
            
            if quality_report['warnings']:
                f.write("\nData Quality Warnings:\n")
                for warning in quality_report['warnings']:
                    f.write(f"  • {warning}\n")
        
        return str(model_path)
    
    def train_regional_model(self, min_samples: int = 50) -> str:
        """Complete regional model training pipeline"""
        
        logger.info(f"Starting regional model training for {self.region}")
        
        # Load regional data
        df = self.load_regional_data()
        
        # Validate data quality
        quality_report = self.validate_data_quality(df)
        
        if quality_report['total_samples'] < min_samples:
            logger.warning(f"Insufficient samples: {quality_report['total_samples']} < {min_samples}")
        
        # Engineer features
        df_processed, feature_columns = self.engineer_regional_features(df)
        
        # Prepare training data
        X = df_processed[feature_columns].values
        
        # Encode targets
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df_processed['risk_level'].dropna())
        
        # Remove rows with missing targets
        valid_indices = df_processed['risk_level'].dropna().index
        X = X[valid_indices]
        
        logger.info(f"Training with {len(X)} samples and {len(feature_columns)} features")
        
        # Train models
        results = self.train_multiple_models(X, y, feature_columns)
        
        if not results:
            raise ValueError("No models were successfully trained")
        
        # Add label encoder to best model results
        best_model_name = results['best_model']
        results[best_model_name]['label_encoder'] = label_encoder
        
        # Evaluate models
        self.evaluate_regional_model(results, quality_report)
        
        # Save best model
        model_path = self.save_regional_model(results, quality_report)
        
        logger.info(f"Regional model training completed: {model_path}")
        return model_path


def train_regional_models_demo():
    """Demonstrate regional model training"""
    
    print("=== Regional Avalanche Model Training Demo ===\n")
    
    # Test different regions
    regions = ["Colorado", "British Columbia"]
    
    for region in regions:
        print(f"\n{'='*60}")
        print(f"TRAINING REGIONAL MODEL - {region.upper()}")
        print('='*60)
        
        try:
            trainer = RegionalModelTrainer(region)
            model_path = trainer.train_regional_model(min_samples=10)  # Lower threshold for demo
            
            print(f"\n✅ Successfully trained {region} regional model")
            print(f"📁 Model saved to: {model_path}")
            
        except Exception as e:
            print(f"\n❌ Error training {region} model: {e}")
            continue


if __name__ == "__main__":
    train_regional_models_demo()