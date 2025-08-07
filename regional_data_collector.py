#!/usr/bin/env python3
"""
Regional Avalanche Data Collection System

This module provides tools to collect real avalanche data from:
1. Avalanche center databases (CAIC, Avalanche Canada, etc.)
2. SNOTEL weather stations
3. Field observation forms
4. Historical avalanche bulletins
5. Research datasets

Author: AI Python Developer
"""

import os
import pandas as pd
import numpy as np
import requests
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import re
import time
from bs4 import BeautifulSoup
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RegionalObservation:
    """Standardized regional avalanche observation"""
    # Location and time
    observation_id: str
    region: str
    location_name: str
    latitude: float
    longitude: float
    elevation: int
    timestamp: datetime
    
    # Observer information
    observer_type: str  # professional, public, automated
    observer_experience: str  # beginner, intermediate, advanced, professional
    
    # Weather conditions
    air_temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[str] = None
    precipitation_24h: Optional[float] = None
    new_snow_24h: Optional[float] = None
    
    # Snowpack
    total_depth: Optional[float] = None
    snow_density: Optional[float] = None
    surface_condition: Optional[str] = None
    
    # Stability indicators
    avalanche_activity: bool = False
    avalanche_size: Optional[str] = None  # D1, D2, D3, D4, D5
    avalanche_type: Optional[str] = None  # SS, WS, WL, etc.
    instability_signs: Optional[str] = None
    
    # Terrain
    slope_angle: Optional[float] = None
    aspect: Optional[str] = None
    terrain_feature: Optional[str] = None
    
    # Assessment
    danger_rating: Optional[str] = None  # Low, Moderate, Considerable, High, Extreme
    confidence: Optional[str] = None  # Low, Moderate, High
    
    # Data source
    data_source: str = "unknown"
    source_url: Optional[str] = None
    
class RegionalDataCollector:
    """Collect avalanche data from multiple regional sources"""
    
    def __init__(self, region: str, output_dir: str = "/workspace/regional_data"):
        self.region = region
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for storing collected data
        self.db_path = self.output_dir / f"{region.lower()}_avalanche_data.db"
        self.setup_database()
        
        # Region-specific configurations
        self.region_configs = {
            "colorado": {
                "caic_zones": ["BTL", "SLV", "EC", "NC", "SC", "SW", "FC", "GJ", "SAG", "VU"],
                "snotel_stations": ["713:CO:SNTL", "771:CO:SNTL", "465:CO:SNTL"],
                "base_url": "https://avalanche.state.co.us"
            },
            "british_columbia": {
                "avalanche_canada_regions": ["sea-to-sky", "south-coast-inland", "north-rockies"],
                "base_url": "https://avalanche.ca"
            },
            "utah": {
                "uac_regions": ["salt-lake", "provo", "ogden", "skyline", "moab"],
                "base_url": "https://utahavalanchecenter.org"
            },
            "washington": {
                "nwac_zones": ["olympic", "hood-canal", "puget-sound", "north-cascades", "south-cascades"],
                "base_url": "https://nwac.us"
            }
        }
        
    def setup_database(self):
        """Setup SQLite database for regional data storage"""
        conn = sqlite3.connect(self.db_path)
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS observations (
            observation_id TEXT PRIMARY KEY,
            region TEXT NOT NULL,
            location_name TEXT,
            latitude REAL,
            longitude REAL,
            elevation INTEGER,
            timestamp DATETIME,
            observer_type TEXT,
            observer_experience TEXT,
            air_temperature REAL,
            wind_speed REAL,
            wind_direction TEXT,
            precipitation_24h REAL,
            new_snow_24h REAL,
            total_depth REAL,
            snow_density REAL,
            surface_condition TEXT,
            avalanche_activity BOOLEAN,
            avalanche_size TEXT,
            avalanche_type TEXT,
            instability_signs TEXT,
            slope_angle REAL,
            aspect TEXT,
            terrain_feature TEXT,
            danger_rating TEXT,
            confidence TEXT,
            data_source TEXT,
            source_url TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        conn.execute(create_table_sql)
        conn.commit()
        conn.close()
        logger.info(f"Database setup completed: {self.db_path}")
    
    def collect_caic_data(self, days_back: int = 30) -> List[RegionalObservation]:
        """
        Collect data from Colorado Avalanche Information Center
        
        Args:
            days_back: Number of days to look back for observations
            
        Returns:
            List of regional observations
        """
        logger.info("Collecting CAIC data...")
        observations = []
        
        if self.region.lower() != "colorado":
            logger.warning("CAIC data collection only available for Colorado region")
            return observations
        
        # Sample CAIC data structure (would be replaced with actual API calls)
        sample_caic_data = [
            {
                "id": "CAIC_001",
                "location": "Loveland Pass",
                "lat": 39.6656,
                "lon": -105.8019,
                "elevation": 3655,
                "date": "2024-01-15",
                "observer": "Professional",
                "new_snow": 25,
                "wind_speed": 35,
                "wind_direction": "SW",
                "danger_rating": "Considerable",
                "avalanche_activity": True,
                "avalanche_size": "D2",
                "slope_angle": 38,
                "aspect": "NE"
            },
            {
                "id": "CAIC_002", 
                "location": "Berthoud Pass",
                "lat": 39.7979,
                "lon": -105.7764,
                "elevation": 3446,
                "date": "2024-01-16",
                "observer": "Public",
                "new_snow": 15,
                "wind_speed": 25,
                "wind_direction": "W",
                "danger_rating": "Moderate",
                "avalanche_activity": False,
                "slope_angle": 32,
                "aspect": "N"
            }
        ]
        
        for data in sample_caic_data:
            obs = RegionalObservation(
                observation_id=data["id"],
                region="Colorado",
                location_name=data["location"],
                latitude=data["lat"],
                longitude=data["lon"],
                elevation=data["elevation"],
                timestamp=datetime.strptime(data["date"], "%Y-%m-%d"),
                observer_type=data["observer"].lower(),
                observer_experience="professional" if data["observer"] == "Professional" else "intermediate",
                new_snow_24h=data.get("new_snow"),
                wind_speed=data.get("wind_speed"),
                wind_direction=data.get("wind_direction"),
                danger_rating=data.get("danger_rating"),
                avalanche_activity=data.get("avalanche_activity", False),
                avalanche_size=data.get("avalanche_size"),
                slope_angle=data.get("slope_angle"),
                aspect=data.get("aspect"),
                data_source="CAIC"
            )
            observations.append(obs)
        
        logger.info(f"Collected {len(observations)} CAIC observations")
        return observations
    
    def collect_avalanche_canada_data(self, days_back: int = 30) -> List[RegionalObservation]:
        """Collect data from Avalanche Canada"""
        logger.info("Collecting Avalanche Canada data...")
        observations = []
        
        if self.region.lower() != "british_columbia":
            logger.warning("Avalanche Canada data collection only available for BC region")
            return observations
        
        # Sample Avalanche Canada data
        sample_ac_data = [
            {
                "id": "AC_001",
                "location": "Whistler Backcountry",
                "lat": 50.1163,
                "lon": -122.9574,
                "elevation": 2200,
                "date": "2024-01-15",
                "observer": "Professional",
                "new_snow": 30,
                "temperature": -8,
                "wind_speed": 40,
                "danger_rating": "High",
                "avalanche_activity": True,
                "avalanche_size": "D3"
            }
        ]
        
        for data in sample_ac_data:
            obs = RegionalObservation(
                observation_id=data["id"],
                region="British Columbia",
                location_name=data["location"],
                latitude=data["lat"],
                longitude=data["lon"], 
                elevation=data["elevation"],
                timestamp=datetime.strptime(data["date"], "%Y-%m-%d"),
                observer_type="professional",
                observer_experience="professional",
                air_temperature=data.get("temperature"),
                new_snow_24h=data.get("new_snow"),
                wind_speed=data.get("wind_speed"),
                danger_rating=data.get("danger_rating"),
                avalanche_activity=data.get("avalanche_activity", False),
                avalanche_size=data.get("avalanche_size"),
                data_source="Avalanche Canada"
            )
            observations.append(obs)
        
        logger.info(f"Collected {len(observations)} Avalanche Canada observations")
        return observations
    
    def collect_snotel_data(self, days_back: int = 30) -> List[RegionalObservation]:
        """Collect automated weather station data from SNOTEL network"""
        logger.info("Collecting SNOTEL data...")
        observations = []
        
        # Sample SNOTEL station data
        sample_snotel_data = [
            {
                "station_id": "713_CO_SNTL",
                "station_name": "Loveland Basin",
                "lat": 39.6833,
                "lon": -105.8833,
                "elevation": 3505,
                "date": "2024-01-15",
                "temperature": -12,
                "precipitation": 2.5,
                "snow_depth": 145,
                "wind_speed": 28
            }
        ]
        
        for data in sample_snotel_data:
            obs = RegionalObservation(
                observation_id=f"SNOTEL_{data['station_id']}_{data['date']}",
                region=self.region,
                location_name=data["station_name"],
                latitude=data["lat"],
                longitude=data["lon"],
                elevation=data["elevation"],
                timestamp=datetime.strptime(data["date"], "%Y-%m-%d"),
                observer_type="automated",
                observer_experience="professional",
                air_temperature=data.get("temperature"),
                precipitation_24h=data.get("precipitation"),
                total_depth=data.get("snow_depth"),
                wind_speed=data.get("wind_speed"),
                data_source="SNOTEL"
            )
            observations.append(obs)
        
        logger.info(f"Collected {len(observations)} SNOTEL observations")
        return observations
    
    def parse_field_observation_form(self, form_data: Dict) -> RegionalObservation:
        """Parse field observation form into standardized format"""
        
        return RegionalObservation(
            observation_id=form_data.get("observation_id", f"FIELD_{int(time.time())}"),
            region=self.region,
            location_name=form_data.get("location_name", "Unknown"),
            latitude=float(form_data.get("latitude", 0)),
            longitude=float(form_data.get("longitude", 0)),
            elevation=int(form_data.get("elevation", 0)),
            timestamp=datetime.fromisoformat(form_data.get("timestamp", datetime.now().isoformat())),
            observer_type=form_data.get("observer_type", "public"),
            observer_experience=form_data.get("observer_experience", "intermediate"),
            air_temperature=form_data.get("air_temperature"),
            wind_speed=form_data.get("wind_speed"),
            wind_direction=form_data.get("wind_direction"),
            precipitation_24h=form_data.get("precipitation_24h"),
            new_snow_24h=form_data.get("new_snow_24h"),
            total_depth=form_data.get("total_depth"),
            snow_density=form_data.get("snow_density"),
            surface_condition=form_data.get("surface_condition"),
            avalanche_activity=form_data.get("avalanche_activity", False),
            avalanche_size=form_data.get("avalanche_size"),
            avalanche_type=form_data.get("avalanche_type"),
            instability_signs=form_data.get("instability_signs"),
            slope_angle=form_data.get("slope_angle"),
            aspect=form_data.get("aspect"),
            terrain_feature=form_data.get("terrain_feature"),
            danger_rating=form_data.get("danger_rating"),
            confidence=form_data.get("confidence"),
            data_source="Field Observation"
        )
    
    def store_observations(self, observations: List[RegionalObservation]):
        """Store observations in the database"""
        conn = sqlite3.connect(self.db_path)
        
        for obs in observations:
            # Convert dataclass to dict and handle datetime
            obs_dict = asdict(obs)
            obs_dict['timestamp'] = obs.timestamp.isoformat()
            
            # Insert or replace observation
            columns = list(obs_dict.keys())
            placeholders = ['?' for _ in columns]
            
            sql = f"""
            INSERT OR REPLACE INTO observations ({','.join(columns)})
            VALUES ({','.join(placeholders)})
            """
            
            conn.execute(sql, list(obs_dict.values()))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(observations)} observations in database")
    
    def collect_all_regional_data(self, days_back: int = 30) -> pd.DataFrame:
        """Collect data from all available regional sources"""
        all_observations = []
        
        # Collect from different sources based on region
        if self.region.lower() == "colorado":
            all_observations.extend(self.collect_caic_data(days_back))
        elif self.region.lower() == "british_columbia":
            all_observations.extend(self.collect_avalanche_canada_data(days_back))
        
        # Always try to collect SNOTEL data (available across regions)
        all_observations.extend(self.collect_snotel_data(days_back))
        
        # Store in database
        if all_observations:
            self.store_observations(all_observations)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(obs) for obs in all_observations])
        
        # Save to CSV
        csv_path = self.output_dir / f"{self.region.lower()}_observations.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} observations to {csv_path}")
        
        return df
    
    def generate_training_dataset(self, min_samples: int = 50) -> pd.DataFrame:
        """Generate training dataset with proper labels and features"""
        
        # Load collected observations
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM observations", conn)
        conn.close()
        
        if len(df) < min_samples:
            logger.warning(f"Only {len(df)} samples available, need at least {min_samples}")
        
        # Convert danger ratings to risk levels for model training
        danger_mapping = {
            "Low": "Stable",
            "Moderate": "Stable", 
            "Considerable": "Borderline",
            "High": "Unstable",
            "Extreme": "Unstable"
        }
        
        df['risk_level'] = df['danger_rating'].map(danger_mapping)
        
        # Filter out rows without risk assessment
        training_df = df[df['risk_level'].notna()].copy()
        
        # Create derived features
        training_df['has_new_snow'] = (training_df['new_snow_24h'] > 0).astype(int)
        training_df['high_wind'] = (training_df['wind_speed'] > 25).astype(int)
        training_df['steep_slope'] = (training_df['slope_angle'] > 35).astype(int)
        training_df['recent_avalanche'] = training_df['avalanche_activity'].astype(int)
        
        # Convert aspects to numerical
        aspect_mapping = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        training_df['aspect_numerical'] = training_df['aspect'].map(aspect_mapping).fillna(0) / 360.0
        
        # Save training dataset
        training_path = self.output_dir / f"{self.region.lower()}_training_data.csv"
        training_df.to_csv(training_path, index=False)
        logger.info(f"Generated training dataset with {len(training_df)} samples: {training_path}")
        
        return training_df
    
    def create_field_observation_template(self) -> Dict:
        """Create a template for field observations"""
        template = {
            "observation_id": "FIELD_YYYY_MM_DD_###",
            "location_name": "Enter location name",
            "latitude": 0.0,
            "longitude": 0.0,
            "elevation": 0,
            "timestamp": datetime.now().isoformat(),
            "observer_type": "public",  # public, professional, automated
            "observer_experience": "intermediate",  # beginner, intermediate, advanced, professional
            
            # Weather (last 24 hours)
            "air_temperature": None,  # Celsius
            "wind_speed": None,  # km/h
            "wind_direction": None,  # N, NE, E, SE, S, SW, W, NW
            "precipitation_24h": None,  # mm
            "new_snow_24h": None,  # cm
            
            # Snowpack
            "total_depth": None,  # cm
            "snow_density": None,  # g/cm³
            "surface_condition": None,  # powder, wind crust, sun crust, etc.
            
            # Avalanche activity
            "avalanche_activity": False,  # True/False
            "avalanche_size": None,  # D1, D2, D3, D4, D5
            "avalanche_type": None,  # SS, WS, WL, etc.
            "instability_signs": None,  # cracking, whumpfing, etc.
            
            # Terrain
            "slope_angle": None,  # degrees
            "aspect": None,  # N, NE, E, SE, S, SW, W, NW
            "terrain_feature": None,  # ridgeline, gully, open slope, etc.
            
            # Assessment
            "danger_rating": None,  # Low, Moderate, Considerable, High, Extreme
            "confidence": None  # Low, Moderate, High
        }
        
        # Save template
        template_path = self.output_dir / "field_observation_template.json"
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2, default=str)
        
        logger.info(f"Field observation template saved to: {template_path}")
        return template


def collect_regional_data_demo():
    """Demonstrate regional data collection"""
    
    print("=== Regional Avalanche Data Collection Demo ===\n")
    
    # Test different regions
    regions = ["Colorado", "British Columbia"]
    
    for region in regions:
        print(f"\n--- Collecting data for {region} ---")
        
        collector = RegionalDataCollector(region)
        
        # Collect regional data
        df = collector.collect_all_regional_data(days_back=30)
        
        print(f"Collected {len(df)} observations")
        print(f"Data sources: {df['data_source'].value_counts().to_dict()}")
        
        if 'danger_rating' in df.columns:
            print(f"Danger ratings: {df['danger_rating'].value_counts().to_dict()}")
        
        # Generate training dataset
        training_df = collector.generate_training_dataset()
        print(f"Training dataset: {len(training_df)} samples")
        
        if 'risk_level' in training_df.columns:
            print(f"Risk levels: {training_df['risk_level'].value_counts().to_dict()}")
        
        # Create field observation template
        template = collector.create_field_observation_template()
        print(f"Field observation template created")


if __name__ == "__main__":
    collect_regional_data_demo()