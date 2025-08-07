#!/usr/bin/env python3
"""
Production Deployment Guide for Avalanche Risk Classification

This script demonstrates how to adapt the pipeline for real-world use with:
1. Real snowpack measurement data
2. Actual snow profile images
3. Production deployment considerations
4. Data collection integration
5. Real-time prediction capabilities

Author: AI Python Developer
"""

import os
import pandas as pd
import numpy as np
import cv2
import joblib
import sqlite3
# import requests  # Commented out for demo - install for production use
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Import our existing pipeline
from avalanche_risk_classifier import AvalancheRiskClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SnowpackMeasurement:
    """Data structure for real snowpack measurements"""
    location_id: str
    timestamp: datetime
    elevation: float
    slope_angle: float
    aspect: str
    
    # Snow characteristics
    total_depth: float
    new_snow_24h: float
    snow_density: float
    temperature_surface: float
    temperature_10cm: float
    
    # Stability tests
    compression_test_score: int
    extended_column_test: str
    shear_strength: float
    
    # Weather data
    wind_speed: float
    wind_direction: str
    precipitation_24h: float
    air_temperature: float
    
    # Crystal analysis
    grain_size: float
    grain_type: str
    hardness: int
    
    # Profile image
    profile_image_path: str
    
    # Human assessment (for training)
    expert_risk_assessment: Optional[str] = None
    avalanche_occurred: Optional[bool] = None

class RealWorldAvalanchePipeline:
    """
    Production-ready avalanche risk assessment pipeline for real data
    """
    
    def __init__(self, model_path: str = None, data_source: str = "local"):
        """
        Initialize the production pipeline
        
        Args:
            model_path: Path to trained model, if None will train new model
            data_source: "local", "api", "database"
        """
        self.model_path = model_path
        self.data_source = data_source
        self.classifier = None
        self.db_connection = None
        
        # Real-world feature mapping
        self.feature_mapping = {
            # Snow characteristics (standardized names)
            'total_depth': 'depth',
            'snow_density': 'density', 
            'temperature_surface': 'temperature',
            'shear_strength': 'shear_strength',
            'grain_size': 'grain_size',
            'hardness': 'hardness',
            'slope_angle': 'slope_angle',
            'wind_speed': 'wind_speed',
            'precipitation_24h': 'precipitation',
            'new_snow_24h': 'new_snow',
            # Additional real features
            'compression_test_score': 'compression_test',
            'air_temperature': 'air_temp',
            'elevation': 'elevation'
        }
        
        if model_path and os.path.exists(model_path):
            self.load_trained_model()
        
    def setup_database(self, db_path: str = "avalanche_data.db"):
        """Setup SQLite database for data storage"""
        self.db_connection = sqlite3.connect(db_path)
        
        # Create tables for real data
        create_measurements_table = """
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            elevation REAL,
            slope_angle REAL,
            aspect TEXT,
            total_depth REAL,
            new_snow_24h REAL,
            snow_density REAL,
            temperature_surface REAL,
            temperature_10cm REAL,
            compression_test_score INTEGER,
            extended_column_test TEXT,
            shear_strength REAL,
            wind_speed REAL,
            wind_direction TEXT,
            precipitation_24h REAL,
            air_temperature REAL,
            grain_size REAL,
            grain_type TEXT,
            hardness INTEGER,
            profile_image_path TEXT,
            expert_risk_assessment TEXT,
            predicted_risk TEXT,
            prediction_confidence REAL,
            avalanche_occurred BOOLEAN
        )
        """
        
        create_locations_table = """
        CREATE TABLE IF NOT EXISTS locations (
            location_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            typical_elevation REAL,
            region TEXT,
            active BOOLEAN DEFAULT TRUE
        )
        """
        
        self.db_connection.execute(create_measurements_table)
        self.db_connection.execute(create_locations_table)
        self.db_connection.commit()
        
        logger.info("Database setup completed")
    
    def collect_weather_data(self, lat: float, lon: float) -> Dict:
        """
        Collect real-time weather data from API
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with weather data
        """
        # Example using OpenWeatherMap API (you'll need an API key)
        # api_key = "your_api_key_here"
        # url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        
        # Mock data for demonstration
        mock_weather = {
            'air_temperature': np.random.normal(-5, 3),
            'wind_speed': np.random.normal(10, 5),
            'wind_direction': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            'precipitation_24h': np.random.exponential(2),
            'humidity': np.random.normal(70, 15)
        }
        
        logger.info(f"Collected weather data for coordinates ({lat}, {lon})")
        return mock_weather
    
    def process_real_snow_profile_image(self, image_path: str) -> np.ndarray:
        """
        Enhanced image processing for real snow profile images
        
        Args:
            image_path: Path to snow profile image
            
        Returns:
            Enhanced feature vector
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return np.zeros(30)  # Expanded feature set
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros(30)
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            features = []
            
            # 1. Enhanced histogram features (16 features)
            hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist_normalized = hist_gray.flatten() / hist_gray.sum()
            features.extend(hist_normalized)
            
            # 2. Edge analysis (3 features)
            # Different edge detection methods
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            
            edge_density_canny = np.sum(edges_canny > 0) / (gray.shape[0] * gray.shape[1])
            edge_density_sobel = np.mean(np.abs(edges_sobel)) / 255.0
            edge_variance = np.var(edges_canny) / 255.0
            
            features.extend([edge_density_canny, edge_density_sobel, edge_variance])
            
            # 3. Texture analysis (4 features)
            # Local Binary Pattern approximation
            mean_intensity = np.mean(gray) / 255.0
            std_intensity = np.std(gray) / 255.0
            skewness = self._calculate_skewness(gray)
            entropy = self._calculate_entropy(gray)
            
            features.extend([mean_intensity, std_intensity, skewness, entropy])
            
            # 4. Layer detection (3 features)
            # Horizontal gradients to detect snow layers
            horizontal_grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            layer_strength = np.mean(np.abs(horizontal_grad)) / 255.0
            
            # Detect horizontal lines (potential layer boundaries)
            lines = cv2.HoughLinesP(edges_canny, 1, np.pi/180, threshold=50, 
                                  minLineLength=50, maxLineGap=10)
            horizontal_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    if angle < 15 or angle > 165:  # Horizontal-ish lines
                        horizontal_lines += 1
            
            layer_count = horizontal_lines / 100.0  # Normalized
            layer_uniformity = 1.0 - (np.std(horizontal_grad) / np.mean(np.abs(horizontal_grad)) if np.mean(np.abs(horizontal_grad)) > 0 else 0)
            
            features.extend([layer_strength, layer_count, layer_uniformity])
            
            # 5. Color analysis for snow quality (4 features)
            # HSV analysis can indicate snow crystal types and conditions
            hue_mean = np.mean(hsv[:,:,0]) / 180.0
            saturation_mean = np.mean(hsv[:,:,1]) / 255.0
            value_mean = np.mean(hsv[:,:,2]) / 255.0
            blue_dominance = np.mean(image[:,:,0]) / 255.0  # Blue channel
            
            features.extend([hue_mean, saturation_mean, value_mean, blue_dominance])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return np.zeros(30)
    
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of image intensities"""
        flat = image.flatten()
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        if std_val == 0:
            return 0
        return np.mean(((flat - mean_val) / std_val) ** 3)
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of image"""
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
        hist = hist[hist > 0]  # Remove zeros
        prob = hist / hist.sum()
        return -np.sum(prob * np.log2(prob))
    
    def standardize_measurement_data(self, measurement: SnowpackMeasurement) -> pd.DataFrame:
        """
        Convert real measurement to standardized format for the model
        
        Args:
            measurement: SnowpackMeasurement object
            
        Returns:
            DataFrame in format expected by model
        """
        # Base features from measurement
        base_features = {
            'depth': measurement.total_depth,
            'density': measurement.snow_density,
            'temperature': measurement.temperature_surface,
            'shear_strength': measurement.shear_strength,
            'grain_size': measurement.grain_size,
            'hardness': measurement.hardness,
            'slope_angle': measurement.slope_angle,
            'wind_speed': measurement.wind_speed,
            'precipitation': measurement.precipitation_24h,
            'new_snow': measurement.new_snow_24h,
            'compression_test': measurement.compression_test_score,
            'air_temp': measurement.air_temperature,
            'elevation': measurement.elevation
        }
        
        # Add derived features that might be important
        derived_features = {
            'temperature_gradient': measurement.temperature_surface - measurement.temperature_10cm,
            'density_ratio': measurement.snow_density / 0.4 if measurement.snow_density else 1.0,  # Ratio to average density
            'load_factor': measurement.new_snow_24h * measurement.snow_density if measurement.new_snow_24h and measurement.snow_density else 0,
            'wind_load': measurement.wind_speed * measurement.new_snow_24h if measurement.wind_speed and measurement.new_snow_24h else 0,
            'aspect_numerical': self._convert_aspect_to_numerical(measurement.aspect),
            'stability_index': self._calculate_stability_index(measurement)
        }
        
        # Combine features
        all_features = {**base_features, **derived_features}
        
        # Fill any missing features to reach 34 total
        remaining_features = 34 - len(all_features)
        for i in range(remaining_features):
            all_features[f'derived_feature_{i+1}'] = 0.0
        
        # Create DataFrame
        df = pd.DataFrame([all_features])
        
        # Add image path
        df['profile_image'] = measurement.profile_image_path
        
        # Add target if available (for training)
        if measurement.expert_risk_assessment:
            df['risk_level'] = measurement.expert_risk_assessment
        
        return df
    
    def _convert_aspect_to_numerical(self, aspect: str) -> float:
        """Convert aspect string to numerical value"""
        aspect_map = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        return aspect_map.get(aspect, 0) / 360.0
    
    def _calculate_stability_index(self, measurement: SnowpackMeasurement) -> float:
        """Calculate a simple stability index from available data"""
        stability = 0.5  # Neutral starting point
        
        # Adjust based on various factors
        if measurement.shear_strength:
            stability += (measurement.shear_strength - 15) / 30.0  # Normalize around 15 kPa
        
        if measurement.compression_test_score:
            stability -= (30 - measurement.compression_test_score) / 30.0  # Lower CT score = less stable
        
        if measurement.new_snow_24h:
            stability -= measurement.new_snow_24h / 50.0  # Recent snow reduces stability
        
        if measurement.wind_speed:
            stability -= (measurement.wind_speed - 10) / 20.0  # High wind reduces stability
        
        return max(0, min(1, stability))  # Clamp between 0 and 1
    
    def predict_avalanche_risk(self, measurement: SnowpackMeasurement) -> Dict:
        """
        Make real-time prediction for a new measurement
        
        Args:
            measurement: SnowpackMeasurement object
            
        Returns:
            Dictionary with prediction results
        """
        if not self.classifier or not self.classifier.is_fitted:
            raise ValueError("Model not loaded or trained. Please train or load a model first.")
        
        # Standardize the measurement data
        df = self.standardize_measurement_data(measurement)
        
        # Extract structured features (exclude image and target columns)
        feature_cols = [col for col in df.columns if col not in ['profile_image', 'risk_level']]
        structured_features = df[feature_cols]
        
        # Extract OpenCV features from image using original method for compatibility
        if hasattr(self.classifier, 'extract_opencv_features'):
            opencv_features = self.classifier.extract_opencv_features(measurement.profile_image_path)
            opencv_features = opencv_features.reshape(1, -1)  # Reshape for single prediction
        else:
            # Fallback to simplified features that match original model (20 features)
            opencv_features = self.process_real_snow_profile_image(measurement.profile_image_path)[:20]
            opencv_features = opencv_features.reshape(1, -1)
        
        # Combine features
        combined_features = np.hstack([structured_features.values, opencv_features])
        
        # Scale features
        scaled_features = self.classifier.scaler.transform(combined_features)
        
        # Make prediction
        prediction = self.classifier.model.predict(scaled_features)[0]
        probabilities = self.classifier.model.predict_proba(scaled_features)[0]
        
        # Get class names
        class_names = self.classifier.label_encoder.classes_
        risk_level = class_names[prediction]
        
        # Calculate confidence
        confidence = np.max(probabilities)
        
        # Get feature importance for this prediction
        feature_importance = self.classifier.model.feature_importances_
        top_features = np.argsort(feature_importance)[-5:]  # Top 5 most important features
        
        result = {
            'risk_level': risk_level,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(class_names, probabilities)
            },
            'top_contributing_factors': [
                self.classifier.feature_names[i] if hasattr(self.classifier, 'feature_names') 
                else f'feature_{i}' 
                for i in top_features
            ],
            'measurement_timestamp': measurement.timestamp.isoformat(),
            'location_id': measurement.location_id
        }
        
        # Store prediction in database if available
        if self.db_connection:
            self._store_prediction(measurement, result)
        
        logger.info(f"Predicted risk level: {risk_level} (confidence: {confidence:.2f})")
        return result
    
    def _store_prediction(self, measurement: SnowpackMeasurement, prediction: Dict):
        """Store measurement and prediction in database"""
        query = """
        INSERT INTO measurements (
            location_id, timestamp, elevation, slope_angle, aspect,
            total_depth, new_snow_24h, snow_density, temperature_surface, temperature_10cm,
            compression_test_score, extended_column_test, shear_strength,
            wind_speed, wind_direction, precipitation_24h, air_temperature,
            grain_size, grain_type, hardness, profile_image_path,
            expert_risk_assessment, predicted_risk, prediction_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.db_connection.execute(query, (
            measurement.location_id, measurement.timestamp, measurement.elevation,
            measurement.slope_angle, measurement.aspect, measurement.total_depth,
            measurement.new_snow_24h, measurement.snow_density, measurement.temperature_surface,
            measurement.temperature_10cm, measurement.compression_test_score,
            measurement.extended_column_test, measurement.shear_strength,
            measurement.wind_speed, measurement.wind_direction, measurement.precipitation_24h,
            measurement.air_temperature, measurement.grain_size, measurement.grain_type,
            measurement.hardness, measurement.profile_image_path, measurement.expert_risk_assessment,
            prediction['risk_level'], prediction['confidence']
        ))
        self.db_connection.commit()
    
    def load_trained_model(self):
        """Load a previously trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model_data = joblib.load(self.model_path)
        self.classifier = AvalancheRiskClassifier("", "")  # Initialize empty
        self.classifier.model = model_data['model']
        self.classifier.scaler = model_data['scaler']
        self.classifier.label_encoder = model_data['label_encoder']
        self.classifier.feature_names = model_data['feature_names']
        self.classifier.is_fitted = True
        
        logger.info(f"Model loaded successfully from {self.model_path}")
    
    def batch_process_measurements(self, measurements: List[SnowpackMeasurement]) -> List[Dict]:
        """Process multiple measurements in batch"""
        results = []
        
        for measurement in measurements:
            try:
                result = self.predict_avalanche_risk(measurement)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing measurement {measurement.location_id}: {e}")
                results.append({
                    'error': str(e),
                    'location_id': measurement.location_id,
                    'timestamp': measurement.timestamp.isoformat()
                })
        
        return results
    
    def generate_risk_report(self, location_id: str, days_back: int = 7) -> Dict:
        """Generate a comprehensive risk report for a location"""
        if not self.db_connection:
            raise ValueError("Database connection required for risk reports")
        
        # Get recent measurements
        query = """
        SELECT * FROM measurements 
        WHERE location_id = ? AND timestamp >= ?
        ORDER BY timestamp DESC
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cursor = self.db_connection.execute(query, (location_id, cutoff_date))
        measurements = cursor.fetchall()
        
        if not measurements:
            return {'error': f'No measurements found for location {location_id}'}
        
        # Analyze trends
        risk_levels = [m[22] for m in measurements if m[22]]  # predicted_risk column
        confidences = [m[23] for m in measurements if m[23]]  # prediction_confidence column
        
        report = {
            'location_id': location_id,
            'report_period': f"{cutoff_date.date()} to {datetime.now().date()}",
            'total_measurements': len(measurements),
            'current_risk_level': risk_levels[0] if risk_levels else 'Unknown',
            'average_confidence': np.mean(confidences) if confidences else 0,
            'risk_trend': self._analyze_risk_trend(risk_levels),
            'recommendations': self._generate_recommendations(measurements[-5:] if len(measurements) >= 5 else measurements)
        }
        
        return report
    
    def _analyze_risk_trend(self, risk_levels: List[str]) -> str:
        """Analyze trend in risk levels"""
        if len(risk_levels) < 2:
            return "Insufficient data"
        
        risk_values = {'Stable': 1, 'Borderline': 2, 'Unstable': 3}
        numeric_risks = [risk_values.get(risk, 2) for risk in risk_levels]
        
        recent_avg = np.mean(numeric_risks[:3]) if len(numeric_risks) >= 3 else numeric_risks[0]
        older_avg = np.mean(numeric_risks[3:]) if len(numeric_risks) > 3 else numeric_risks[-1]
        
        if recent_avg > older_avg + 0.5:
            return "Increasing risk"
        elif recent_avg < older_avg - 0.5:
            return "Decreasing risk"
        else:
            return "Stable trend"
    
    def _generate_recommendations(self, recent_measurements) -> List[str]:
        """Generate safety recommendations based on recent data"""
        recommendations = []
        
        if not recent_measurements:
            return ["No recent data available for recommendations"]
        
        latest = recent_measurements[0]
        predicted_risk = latest[22] if latest[22] else "Unknown"
        
        if predicted_risk == "Unstable":
            recommendations.extend([
                "HIGH RISK: Avoid steep slopes and wind-loaded areas",
                "Consider postponing backcountry activities",
                "If travel necessary, stick to low-angle terrain (<30°)",
                "Travel one at a time through avalanche terrain"
            ])
        elif predicted_risk == "Borderline":
            recommendations.extend([
                "MODERATE RISK: Exercise heightened caution",
                "Avoid obvious avalanche terrain",
                "Choose conservative route options",
                "Carry full avalanche safety equipment"
            ])
        else:  # Stable
            recommendations.extend([
                "LOWER RISK: Normal precautions apply",
                "Continue to assess conditions throughout the day",
                "Carry standard avalanche safety equipment"
            ])
        
        return recommendations


def example_real_world_usage():
    """
    Example of how to use the pipeline with real data
    """
    print("=== Real-World Avalanche Risk Assessment Example ===\n")
    
    # Initialize the production pipeline
    pipeline = RealWorldAvalanchePipeline(
        model_path="/workspace/avalanche_model.pkl",
        data_source="local"
    )
    
    # Setup database for storing measurements
    pipeline.setup_database()
    
    # Example real measurement (this would come from field observations)
    measurement = SnowpackMeasurement(
        location_id="WHISTLER_PEAK_001",
        timestamp=datetime.now(),
        elevation=2200.0,
        slope_angle=38.0,
        aspect="NE",
        total_depth=165.0,
        new_snow_24h=25.0,
        snow_density=0.45,
        temperature_surface=-8.0,
        temperature_10cm=-5.0,
        compression_test_score=18,
        extended_column_test="ECTPV",
        shear_strength=12.5,
        wind_speed=25.0,
        wind_direction="SW",
        precipitation_24h=35.0,
        air_temperature=-12.0,
        grain_size=1.8,
        grain_type="rounded",
        hardness=3,
        profile_image_path="/workspace/sample_images/profile_0001.jpg",
        expert_risk_assessment="Unstable"  # For training data
    )
    
    print("Processing real snowpack measurement...")
    print(f"Location: {measurement.location_id}")
    print(f"Timestamp: {measurement.timestamp}")
    print(f"Conditions: {measurement.new_snow_24h}cm new snow, wind {measurement.wind_speed}km/h")
    print()
    
    # Make prediction
    try:
        result = pipeline.predict_avalanche_risk(measurement)
        
        print("=== PREDICTION RESULTS ===")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Probabilities:")
        for risk, prob in result['probabilities'].items():
            print(f"  {risk}: {prob:.1%}")
        print(f"Top Contributing Factors: {', '.join(result['top_contributing_factors'])}")
        print()
        
        # Generate risk report
        print("=== RISK REPORT ===")
        report = pipeline.generate_risk_report(measurement.location_id, days_back=7)
        print(f"Current Risk: {report.get('current_risk_level', 'Unknown')}")
        print(f"Trend: {report.get('risk_trend', 'Unknown')}")
        print("Recommendations:")
        for rec in report.get('recommendations', []):
            print(f"  • {rec}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Note: This example uses the sample model. For production use,")
        print("train the model with real avalanche data.")


if __name__ == "__main__":
    example_real_world_usage()