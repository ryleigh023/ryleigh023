#!/usr/bin/env python3
"""
Incremental Deployment System for Regional Avalanche Models

This system provides:
1. Model validation against historical data
2. A/B testing framework for gradual rollout
3. Performance monitoring and alerts
4. Rollback capabilities
5. User feedback integration

Author: AI Python Developer
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify, render_template_string
import threading
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import our systems
from regional_model_trainer import RegionalModelTrainer
from production_deployment_guide import RealWorldAvalanchePipeline, SnowpackMeasurement

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Results from model validation"""
    model_name: str
    validation_accuracy: float
    confusion_matrix: List[List[int]]
    classification_report: str
    historical_performance: Dict
    validation_date: datetime
    sample_count: int
    warnings: List[str]

@dataclass
class DeploymentStage:
    """Deployment stage configuration"""
    stage_name: str
    traffic_percentage: float
    success_criteria: Dict
    max_duration_hours: int
    monitoring_metrics: List[str]

@dataclass
class PredictionLog:
    """Log entry for model predictions"""
    prediction_id: str
    timestamp: datetime
    model_version: str
    input_features: Dict
    prediction: str
    confidence: float
    actual_outcome: Optional[str] = None
    user_feedback: Optional[str] = None
    processing_time_ms: float = 0.0

class ModelValidator:
    """Validate models against historical data and known outcomes"""
    
    def __init__(self, data_dir: str = "/workspace/regional_data"):
        self.data_dir = Path(data_dir)
        self.validation_db = self.data_dir / "validation_results.db"
        self.setup_validation_db()
    
    def setup_validation_db(self):
        """Setup database for validation results"""
        conn = sqlite3.connect(self.validation_db)
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS validation_results (
            validation_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            region TEXT NOT NULL,
            validation_date DATETIME,
            accuracy REAL,
            sample_count INTEGER,
            confusion_matrix TEXT,
            classification_report TEXT,
            warnings TEXT,
            passed BOOLEAN
        )
        """
        
        conn.execute(create_table_sql)
        conn.commit()
        conn.close()
    
    def validate_against_historical_data(self, model_path: str, region: str, 
                                       validation_period_days: int = 90) -> ValidationResult:
        """Validate model against historical avalanche data with known outcomes"""
        
        logger.info(f"Validating model against {validation_period_days} days of historical data")
        
        # Load the model
        model_data = joblib.load(model_path)
        
        # Load historical validation data (this would come from avalanche bulletins)
        historical_data = self._load_historical_validation_data(region, validation_period_days)
        
        if len(historical_data) < 10:
            logger.warning(f"Insufficient historical data for validation: {len(historical_data)} samples")
            return ValidationResult(
                model_name=model_data.get('model_type', 'unknown'),
                validation_accuracy=0.0,
                confusion_matrix=[],
                classification_report="Insufficient data",
                historical_performance={},
                validation_date=datetime.now(),
                sample_count=len(historical_data),
                warnings=["Insufficient historical data for validation"]
            )
        
        # Prepare features for validation
        feature_names = model_data['feature_names']
        X_val = []
        y_true = []
        
        for _, row in historical_data.iterrows():
            try:
                # Extract features in the same order as training
                features = []
                for feature_name in feature_names:
                    if feature_name in row:
                        features.append(row[feature_name])
                    else:
                        features.append(0.0)  # Default value for missing features
                
                X_val.append(features)
                y_true.append(row['actual_risk_level'])
                
            except Exception as e:
                logger.warning(f"Error processing validation row: {e}")
                continue
        
        X_val = np.array(X_val)
        
        # Scale features using the model's scaler
        X_val_scaled = model_data['scaler'].transform(X_val)
        
        # Make predictions
        predictions = model_data['model'].predict(X_val_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        cm = confusion_matrix(y_true, predictions)
        cr = classification_report(y_true, predictions, output_dict=True)
        
        # Historical performance analysis
        historical_performance = self._analyze_historical_performance(historical_data, predictions, y_true)
        
        # Generate warnings
        warnings = []
        if accuracy < 0.6:
            warnings.append(f"Low validation accuracy: {accuracy:.3f}")
        if len(set(predictions)) < 3:
            warnings.append("Model not predicting all risk classes")
        
        result = ValidationResult(
            model_name=model_data.get('model_type', 'unknown'),
            validation_accuracy=accuracy,
            confusion_matrix=cm.tolist(),
            classification_report=json.dumps(cr),
            historical_performance=historical_performance,
            validation_date=datetime.now(),
            sample_count=len(y_true),
            warnings=warnings
        )
        
        # Store validation results
        self._store_validation_result(result, region)
        
        logger.info(f"Validation completed - Accuracy: {accuracy:.3f}, Samples: {len(y_true)}")
        return result
    
    def _load_historical_validation_data(self, region: str, days_back: int) -> pd.DataFrame:
        """Load historical data with known outcomes for validation"""
        
        # In a real implementation, this would load from:
        # - Avalanche bulletin archives with forecasts vs actual events
        # - Professional observer reports with subsequent verification
        # - Research datasets with ground truth
        
        # For demo, create synthetic historical validation data
        np.random.seed(42)
        n_samples = min(50, days_back)  # Realistic number of validation samples
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days_back),
            end=datetime.now() - timedelta(days=1),
            periods=n_samples
        )
        
        historical_data = []
        for date in dates:
            # Simulate realistic avalanche scenarios
            elevation = np.random.normal(2500, 500)
            new_snow = np.random.exponential(10)
            wind_speed = np.random.normal(20, 10)
            air_temp = np.random.normal(-8, 5)
            
            # Simulate risk assessment (this would be from bulletins)
            risk_score = (
                new_snow * 0.3 + 
                wind_speed * 0.2 + 
                (elevation - 2000) * 0.001 +
                (-air_temp) * 0.1 +
                np.random.normal(0, 2)
            )
            
            if risk_score < 8:
                actual_risk = 0  # Stable
            elif risk_score < 15:
                actual_risk = 1  # Borderline  
            else:
                actual_risk = 2  # Unstable
            
            # Add some features that match the model's expected inputs
            record = {
                'date': date,
                'elevation': elevation,
                'new_snow_24h': new_snow,
                'wind_speed': wind_speed,
                'air_temperature': air_temp,
                'slope_angle': np.random.normal(35, 10),
                'aspect_numerical': np.random.uniform(0, 1),
                'instability_score': risk_score / 20.0,
                'elevation_normalized': elevation / 4000.0,
                'wind_load': wind_speed * new_snow,
                'actual_risk_level': actual_risk
            }
            
            historical_data.append(record)
        
        return pd.DataFrame(historical_data)
    
    def _analyze_historical_performance(self, historical_data: pd.DataFrame, 
                                      predictions: np.ndarray, y_true: List) -> Dict:
        """Analyze performance patterns in historical data"""
        
        analysis = {
            'accuracy_by_month': {},
            'accuracy_by_risk_level': {},
            'false_negative_rate': 0.0,
            'false_positive_rate': 0.0
        }
        
        # Accuracy by month
        historical_data['month'] = pd.to_datetime(historical_data['date']).dt.month
        for month in historical_data['month'].unique():
            month_mask = historical_data['month'] == month
            if np.sum(month_mask) > 0:
                month_accuracy = accuracy_score(
                    np.array(y_true)[month_mask], 
                    predictions[month_mask]
                )
                analysis['accuracy_by_month'][int(month)] = float(month_accuracy)
        
        # Accuracy by risk level
        for risk_level in [0, 1, 2]:
            level_mask = np.array(y_true) == risk_level
            if np.sum(level_mask) > 0:
                level_accuracy = accuracy_score(
                    np.array(y_true)[level_mask],
                    predictions[level_mask]
                )
                analysis['accuracy_by_risk_level'][risk_level] = float(level_accuracy)
        
        # Safety-critical metrics (false negatives for high risk)
        high_risk_actual = np.array(y_true) == 2  # Unstable
        high_risk_predicted = predictions == 2
        
        if np.sum(high_risk_actual) > 0:
            false_negative_rate = np.sum(high_risk_actual & ~high_risk_predicted) / np.sum(high_risk_actual)
            analysis['false_negative_rate'] = float(false_negative_rate)
        
        return analysis
    
    def _store_validation_result(self, result: ValidationResult, region: str):
        """Store validation result in database"""
        conn = sqlite3.connect(self.validation_db)
        
        validation_id = f"{region}_{result.model_name}_{int(result.validation_date.timestamp())}"
        
        sql = """
        INSERT OR REPLACE INTO validation_results 
        (validation_id, model_name, region, validation_date, accuracy, sample_count,
         confusion_matrix, classification_report, warnings, passed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        passed = result.validation_accuracy > 0.6 and len(result.warnings) == 0
        
        conn.execute(sql, (
            validation_id, result.model_name, region, result.validation_date.isoformat(),
            result.validation_accuracy, result.sample_count,
            json.dumps(result.confusion_matrix), result.classification_report,
            json.dumps(result.warnings), passed
        ))
        
        conn.commit()
        conn.close()

class IncrementalDeployment:
    """Manage incremental deployment with A/B testing and monitoring"""
    
    def __init__(self, region: str, data_dir: str = "/workspace/regional_data"):
        self.region = region
        self.data_dir = Path(data_dir)
        self.deployment_db = self.data_dir / "deployment_logs.db"
        self.setup_deployment_db()
        
        # Deployment configuration
        self.deployment_stages = [
            DeploymentStage(
                stage_name="canary",
                traffic_percentage=5.0,
                success_criteria={"accuracy": 0.65, "response_time_ms": 1000},
                max_duration_hours=24,
                monitoring_metrics=["accuracy", "response_time", "error_rate"]
            ),
            DeploymentStage(
                stage_name="pilot", 
                traffic_percentage=25.0,
                success_criteria={"accuracy": 0.7, "response_time_ms": 800},
                max_duration_hours=72,
                monitoring_metrics=["accuracy", "response_time", "error_rate", "user_feedback"]
            ),
            DeploymentStage(
                stage_name="full",
                traffic_percentage=100.0,
                success_criteria={"accuracy": 0.75, "response_time_ms": 500},
                max_duration_hours=168,  # 1 week
                monitoring_metrics=["accuracy", "response_time", "error_rate", "user_feedback", "safety_metrics"]
            )
        ]
        
        self.current_stage = 0
        self.models = {}  # Store multiple model versions
        self.monitoring_data = []
        
    def setup_deployment_db(self):
        """Setup database for deployment logging"""
        conn = sqlite3.connect(self.deployment_db)
        
        create_tables = [
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                prediction_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                model_version TEXT,
                input_features TEXT,
                prediction TEXT,
                confidence REAL,
                processing_time_ms REAL,
                actual_outcome TEXT,
                user_feedback TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS deployment_stages (
                stage_id TEXT PRIMARY KEY,
                stage_name TEXT,
                start_time DATETIME,
                end_time DATETIME,
                traffic_percentage REAL,
                success_metrics TEXT,
                status TEXT
            )
            """
        ]
        
        for sql in create_tables:
            conn.execute(sql)
        
        conn.commit()
        conn.close()
    
    def load_model_version(self, model_path: str, version_name: str):
        """Load a model version for deployment"""
        logger.info(f"Loading model version: {version_name}")
        
        model_data = joblib.load(model_path)
        
        # Wrap model for production use
        pipeline = RealWorldAvalanchePipeline()
        pipeline.classifier = type('ModelWrapper', (), {})()
        pipeline.classifier.model = model_data['model']
        pipeline.classifier.scaler = model_data['scaler']
        pipeline.classifier.feature_names = model_data['feature_names']
        pipeline.classifier.is_fitted = True
        
        self.models[version_name] = {
            'pipeline': pipeline,
            'metadata': model_data,
            'load_time': datetime.now()
        }
        
        logger.info(f"Model {version_name} loaded successfully")
    
    def route_prediction_request(self, measurement: SnowpackMeasurement) -> Tuple[str, Dict]:
        """Route prediction request to appropriate model version based on traffic split"""
        
        current_stage = self.deployment_stages[self.current_stage]
        
        # Simple traffic routing (in production, use more sophisticated routing)
        if np.random.random() * 100 < current_stage.traffic_percentage:
            model_version = "new_model"
        else:
            model_version = "baseline_model"
        
        # Fallback to available model
        if model_version not in self.models:
            model_version = list(self.models.keys())[0] if self.models else None
        
        if not model_version:
            raise ValueError("No models available for prediction")
        
        # Make prediction
        start_time = time.time()
        try:
            pipeline = self.models[model_version]['pipeline']
            result = pipeline.predict_avalanche_risk(measurement)
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Log prediction
            self._log_prediction(measurement, result, model_version, processing_time)
            
            return model_version, result
            
        except Exception as e:
            logger.error(f"Prediction error with {model_version}: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            # Try fallback model
            fallback_version = "baseline_model" if model_version != "baseline_model" else None
            if fallback_version and fallback_version in self.models:
                logger.info(f"Using fallback model: {fallback_version}")
                result = self.models[fallback_version]['pipeline'].predict_avalanche_risk(measurement)
                self._log_prediction(measurement, result, fallback_version, processing_time, error=str(e))
                return fallback_version, result
            else:
                raise e
    
    def _log_prediction(self, measurement: SnowpackMeasurement, result: Dict, 
                       model_version: str, processing_time: float, error: str = None):
        """Log prediction for monitoring and analysis"""
        
        log_entry = PredictionLog(
            prediction_id=f"{measurement.location_id}_{int(time.time())}",
            timestamp=datetime.now(),
            model_version=model_version,
            input_features=asdict(measurement),
            prediction=result.get('risk_level', 'unknown'),
            confidence=result.get('confidence', 0.0),
            processing_time_ms=processing_time
        )
        
        # Store in database
        conn = sqlite3.connect(self.deployment_db)
        
        sql = """
        INSERT INTO prediction_logs 
        (prediction_id, timestamp, model_version, input_features, prediction, 
         confidence, processing_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        conn.execute(sql, (
            log_entry.prediction_id, log_entry.timestamp.isoformat(),
            log_entry.model_version, json.dumps(log_entry.input_features, default=str),
            log_entry.prediction, log_entry.confidence, log_entry.processing_time_ms
        ))
        
        conn.commit()
        conn.close()
        
        # Add to monitoring data
        self.monitoring_data.append(log_entry)
    
    def monitor_deployment_performance(self) -> Dict:
        """Monitor current deployment stage performance"""
        
        # Analyze recent predictions (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_logs = [log for log in self.monitoring_data if log.timestamp > cutoff_time]
        
        if not recent_logs:
            return {"status": "insufficient_data", "message": "No recent predictions to analyze"}
        
        # Calculate metrics
        metrics = {
            "total_predictions": len(recent_logs),
            "model_versions": {},
            "avg_processing_time": np.mean([log.processing_time_ms for log in recent_logs]),
            "avg_confidence": np.mean([log.confidence for log in recent_logs]),
            "prediction_distribution": {},
            "performance_trends": {}
        }
        
        # Analyze by model version
        for log in recent_logs:
            version = log.model_version
            if version not in metrics["model_versions"]:
                metrics["model_versions"][version] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "avg_processing_time": 0
                }
            
            metrics["model_versions"][version]["count"] += 1
        
        # Calculate averages
        for version, data in metrics["model_versions"].items():
            version_logs = [log for log in recent_logs if log.model_version == version]
            data["avg_confidence"] = np.mean([log.confidence for log in version_logs])
            data["avg_processing_time"] = np.mean([log.processing_time_ms for log in version_logs])
        
        # Prediction distribution
        for log in recent_logs:
            pred = log.prediction
            metrics["prediction_distribution"][pred] = metrics["prediction_distribution"].get(pred, 0) + 1
        
        # Check success criteria for current stage
        current_stage = self.deployment_stages[self.current_stage]
        success_criteria_met = True
        
        if "response_time_ms" in current_stage.success_criteria:
            if metrics["avg_processing_time"] > current_stage.success_criteria["response_time_ms"]:
                success_criteria_met = False
        
        metrics["stage_status"] = {
            "current_stage": current_stage.stage_name,
            "criteria_met": success_criteria_met,
            "traffic_percentage": current_stage.traffic_percentage
        }
        
        return metrics
    
    def advance_deployment_stage(self) -> bool:
        """Advance to next deployment stage if criteria are met"""
        
        performance = self.monitor_deployment_performance()
        
        if performance["stage_status"]["criteria_met"] and self.current_stage < len(self.deployment_stages) - 1:
            self.current_stage += 1
            logger.info(f"Advanced to deployment stage: {self.deployment_stages[self.current_stage].stage_name}")
            return True
        
        return False
    
    def rollback_deployment(self, reason: str):
        """Rollback to previous stable version"""
        logger.warning(f"Rolling back deployment: {reason}")
        
        # Reset to baseline model (100% traffic)
        self.current_stage = 0
        
        # Log rollback event
        conn = sqlite3.connect(self.deployment_db)
        sql = """
        INSERT INTO deployment_stages 
        (stage_id, stage_name, start_time, status)
        VALUES (?, ?, ?, ?)
        """
        
        stage_id = f"rollback_{int(time.time())}"
        conn.execute(sql, (stage_id, "rollback", datetime.now().isoformat(), f"rollback: {reason}"))
        conn.commit()
        conn.close()


# Flask app for monitoring dashboard
app = Flask(__name__)

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Avalanche Model Deployment Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }
        .alert { color: red; font-weight: bold; }
        .success { color: green; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>🏔️ Avalanche Model Deployment Dashboard</h1>
    
    <div id="status">
        <h2>Current Status</h2>
        <div class="metric">
            <strong>Deployment Stage:</strong> {{ metrics.stage_status.current_stage }}
        </div>
        <div class="metric">
            <strong>Traffic Split:</strong> {{ metrics.stage_status.traffic_percentage }}%
        </div>
        <div class="metric {{ 'success' if metrics.stage_status.criteria_met else 'alert' }}">
            <strong>Criteria Met:</strong> {{ 'Yes' if metrics.stage_status.criteria_met else 'No' }}
        </div>
    </div>
    
    <div id="performance">
        <h2>Performance Metrics (24h)</h2>
        <div class="metric">
            <strong>Total Predictions:</strong> {{ metrics.total_predictions }}
        </div>
        <div class="metric">
            <strong>Avg Processing Time:</strong> {{ "%.1f"|format(metrics.avg_processing_time) }}ms
        </div>
        <div class="metric">
            <strong>Avg Confidence:</strong> {{ "%.1%"|format(metrics.avg_confidence) }}
        </div>
    </div>
    
    <div id="models">
        <h2>Model Versions</h2>
        <table>
            <tr>
                <th>Version</th>
                <th>Predictions</th>
                <th>Avg Confidence</th>
                <th>Avg Processing Time</th>
            </tr>
            {% for version, data in metrics.model_versions.items() %}
            <tr>
                <td>{{ version }}</td>
                <td>{{ data.count }}</td>
                <td>{{ "%.1%"|format(data.avg_confidence) }}</td>
                <td>{{ "%.1f"|format(data.avg_processing_time) }}ms</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div id="predictions">
        <h2>Prediction Distribution</h2>
        <table>
            <tr>
                <th>Risk Level</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {% for pred, count in metrics.prediction_distribution.items() %}
            <tr>
                <td>{{ pred }}</td>
                <td>{{ count }}</td>
                <td>{{ "%.1%"|format(count / metrics.total_predictions) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function(){ location.reload(); }, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Deployment monitoring dashboard"""
    # This would connect to your deployment system
    # For demo, return sample metrics
    sample_metrics = {
        "stage_status": {
            "current_stage": "pilot",
            "criteria_met": True,
            "traffic_percentage": 25.0
        },
        "total_predictions": 150,
        "avg_processing_time": 245.3,
        "avg_confidence": 0.78,
        "model_versions": {
            "baseline_model": {"count": 113, "avg_confidence": 0.76, "avg_processing_time": 230.1},
            "new_model": {"count": 37, "avg_confidence": 0.82, "avg_processing_time": 289.7}
        },
        "prediction_distribution": {
            "Stable": 89,
            "Borderline": 42,
            "Unstable": 19
        }
    }
    
    return render_template_string(DASHBOARD_TEMPLATE, metrics=sample_metrics)

def run_incremental_deployment_demo():
    """Demonstrate the complete incremental deployment system"""
    
    print("=== Incremental Deployment Demo ===\n")
    
    region = "Colorado"
    
    # 1. Collect regional data
    print("📊 Step 1: Collecting regional data...")
    from regional_data_collector import RegionalDataCollector
    collector = RegionalDataCollector(region)
    df = collector.collect_all_regional_data()
    training_df = collector.generate_training_dataset()
    print(f"✅ Collected {len(training_df)} training samples")
    
    # 2. Train regional model
    print("\n🎓 Step 2: Training regional model...")
    trainer = RegionalModelTrainer(region)
    model_path = trainer.train_regional_model(min_samples=5)  # Low threshold for demo
    print(f"✅ Model trained and saved: {model_path}")
    
    # 3. Validate model
    print("\n🔍 Step 3: Validating model against historical data...")
    validator = ModelValidator()
    validation_result = validator.validate_against_historical_data(model_path, region)
    print(f"✅ Validation accuracy: {validation_result.validation_accuracy:.3f}")
    if validation_result.warnings:
        print(f"⚠️  Warnings: {', '.join(validation_result.warnings)}")
    
    # 4. Setup incremental deployment
    print("\n📱 Step 4: Setting up incremental deployment...")
    deployment = IncrementalDeployment(region)
    
    # Load baseline and new model (for demo, use same model as both)
    deployment.load_model_version(model_path, "baseline_model")
    deployment.load_model_version(model_path, "new_model")
    
    print("✅ Deployment system initialized")
    
    # 5. Simulate some predictions
    print("\n🔮 Step 5: Simulating predictions...")
    
    # Create sample measurements
    sample_measurements = [
        SnowpackMeasurement(
            location_id=f"TEST_LOC_{i}",
            timestamp=datetime.now(),
            elevation=2200 + i*100,
            slope_angle=35 + i*2,
            aspect="NE",
            total_depth=150 + i*10,
            new_snow_24h=15 + i*2,
            snow_density=0.4 + i*0.05,
            temperature_surface=-8 + i,
            temperature_10cm=-5 + i,
            compression_test_score=20 - i,
            extended_column_test="ECTPV",
            shear_strength=15 - i,
            wind_speed=25 + i*3,
            wind_direction="SW",
            precipitation_24h=20,
            air_temperature=-10 + i,
            grain_size=1.5 + i*0.1,
            grain_type="rounded",
            hardness=3,
            profile_image_path="/workspace/sample_images/profile_0001.jpg"
        )
        for i in range(10)
    ]
    
    # Make predictions through deployment system
    prediction_results = []
    for measurement in sample_measurements:
        try:
            model_version, result = deployment.route_prediction_request(measurement)
            prediction_results.append((model_version, result))
            print(f"  {measurement.location_id}: {result['risk_level']} (confidence: {result['confidence']:.2f}, model: {model_version})")
        except Exception as e:
            print(f"  Error predicting {measurement.location_id}: {e}")
    
    # 6. Monitor performance
    print("\n📈 Step 6: Monitoring deployment performance...")
    performance = deployment.monitor_deployment_performance()
    print(f"✅ Performance monitoring:")
    print(f"  Total predictions: {performance.get('total_predictions', 0)}")
    print(f"  Average confidence: {performance.get('avg_confidence', 0):.2f}")
    print(f"  Current stage: {performance.get('stage_status', {}).get('current_stage', 'unknown')}")
    
    print("\n🎉 Incremental deployment demo completed!")
    print("\n💡 Next steps for production:")
    print("  1. Collect 100+ real regional observations")
    print("  2. Train model with historical avalanche data")
    print("  3. Validate against known outcomes") 
    print("  4. Start with canary deployment (5% traffic)")
    print("  5. Monitor and gradually increase traffic")
    print("  6. Set up automated monitoring and alerts")


if __name__ == "__main__":
    run_incremental_deployment_demo()