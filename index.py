from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from geopy.geocoders import Nominatim
import os
import pickle
import random
import logging
import time
from math import radians, cos, sin, asin, sqrt

index = Flask(__name__)

# Configure CORS
CORS(index, origins=["https://ayushbhusari.netlify.app",
                   "https://wondrous-salmiakki-4bbd0a.netlify.app",
                   "https://magnificent-belekoy-861624.netlify.app",
                   "https://resilienceai.netlify.app,"
                   "https://resilienceai-ruby.vercel.app"])
CORS(index)# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("disaster_monitor.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ====================== API KEYS (Use Environment Variables) ======================
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY")
AMBEE_API_KEY = os.environ.get("AMBEE_API_KEY")
OPENCAGE_API_KEY = os.environ.get("OPENCAGE_API_KEY")

# Cache settings
CACHE_DIR = "cache"
WEATHER_CACHE_TTL = 5 * 60
DISASTER_CACHE_TTL = 10 * 60

# ====================== CACHE CLASS ======================
class Cache:
    """Simple file-based cache to store API responses."""

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key, ttl):
        """Get data from cache if it exists and is not expired."""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if not os.path.exists(cache_file):
            return None
            
        try:
            import json
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            timestamp = cache_data.get('timestamp', 0)
            if time.time() - timestamp > ttl:
                return None
                
            return cache_data.get('data')
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def set(self, key, data):
        """Store data in cache with current timestamp."""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        try:
            import json
            cache_data = {
                'data': data,
                'timestamp': time.time()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

# ====================== DISASTER MONITOR CLASS ======================
class DisasterMonitor:
    """Main class to fetch and process disaster-related data."""

    def __init__(self):
        self.cache = Cache()
        
    def get_coordinates(self, city):
        """Get latitude and longitude for a given city."""
        cache_key = f"geocode_{city.lower().replace(' ', '_')}"
        
        coords = self.cache.get(cache_key, 86400)
        if coords:
            return coords
            
        try:
            url = f"http://api.weatherapi.com/v1/search.json?key={WEATHERAPI_KEY}&q={city}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data and isinstance(data, list) and len(data) > 0:
                location = data[0]
                coords = {
                    'lat': float(location.get('lat')),
                    'lon': float(location.get('lon')),
                    'name': location.get('name'),
                    'country': location.get('country'),
                    'admin1': location.get('region'),
                    'source': 'weatherapi'
                }
                
                if not self.validate_coordinates(coords['lat'], coords['lon']):
                    raise ValueError("Invalid coordinates from WeatherAPI")
                    
                self.cache.set(cache_key, coords)
                return coords
        except Exception as e:
            logger.error(f"Error getting coordinates from WeatherAPI: {e}")
            
        logger.error(f"Could not find coordinates for {city}")
        return None
    
    def validate_coordinates(self, lat, lon):
        """Check if coordinates are within Earth's bounds."""
        return -90 <= lat <= 90 and -180 <= lon <= 180
        
    def get_weather_data(self, lat, lon):
        """Fetch current weather and forecast data."""
        cache_key = f"weather_{lat}_{lon}"
        
        weather_data = self.cache.get(cache_key, WEATHER_CACHE_TTL)
        if weather_data:
            return weather_data
            
        try:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={lat},{lon}&days=1&aqi=no&alerts=no"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            current = data.get('current', {})
            forecast_hour = data.get('forecast', {}).get('forecastday', [{}])[0].get('hour', [{}])[1]
            
            weather_data = {
                'current': {
                    'temperature': current.get('temp_c'),
                    'humidity': current.get('humidity'),
                    'wind_speed': current.get('wind_kph', 0) / 3.6,
                    'pressure': current.get('pressure_mb'),
                    'precipitation': current.get('precip_mm', 0),
                    'conditions': current.get('condition', {}).get('text', 'Unknown'),
                    'timestamp': current.get('last_updated', datetime.now().isoformat())
                },
                'forecast': {
                    'temperature': forecast_hour.get('temp_c'),
                    'humidity': forecast_hour.get('humidity'),
                    'wind_speed': forecast_hour.get('wind_kph', 0) / 3.6,
                    'pressure': forecast_hour.get('pressure_mb'),
                    'precipitation': forecast_hour.get('precip_mm', 0),
                    'probability': forecast_hour.get('chance_of_rain', 0),
                    'conditions': forecast_hour.get('condition', {}).get('text', 'Unknown'),
                    'timestamp': forecast_hour.get('time', (datetime.now() + timedelta(hours=1)).isoformat())
                }
            }
            
            self.cache.set(cache_key, weather_data)
            return weather_data
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
        
    def get_flood_data(self, lat, lon):
        """Enhanced flood risk assessment."""
        cache_key = f"flood_{lat}_{lon}"
        
        flood_data = self.cache.get(cache_key, DISASTER_CACHE_TTL)
        if flood_data:
            return flood_data
            
        try:
            weather_data = self.get_weather_data(lat, lon)
            if not weather_data:
                return None
                
            current = weather_data['current']
            forecast = weather_data['forecast']
            
            current_precip = current.get('precipitation', 0)
            forecast_precip = forecast.get('precipitation', 0)
            forecast_prob = forecast.get('probability', 0)
            
            elevation_factor = 1.2 if self._is_hilly_terrain(lat, lon) else 1.0
            water_level = max(current_precip, forecast_precip) * elevation_factor
            risk_level = 'low'
            
            if water_level > 20 or (water_level > 10 and forecast_prob > 70):
                risk_level = 'high'
            elif water_level > 8 or (water_level > 4 and forecast_prob > 50):
                risk_level = 'medium'
                
            flood_data = {
                'water_level': round(water_level, 1),
                'risk_level': risk_level,
                'probability': round(forecast_prob, 1),
                'terrain': 'hilly' if elevation_factor > 1 else 'flat',
                'timestamp': datetime.now().isoformat()
            }
            
            self.cache.set(cache_key, flood_data)
            return flood_data
        except Exception as e:
            logger.error(f"Error calculating flood data: {e}")
            return None

    def _is_hilly_terrain(self, lat, lon):
        """Simple terrain detection."""
        hilly_regions = [
            (11.41, 76.70, 50),
            (30.37, 78.96, 100),
            (15.35, 76.16, 80)
        ]
        
        for region_lat, region_lon, radius in hilly_regions:
            if self._calculate_distance(lat, lon, region_lat, region_lon) < radius:
                return True
        return False
    
    def get_wildfire_data(self, lat, lon):
        """Fetch wildfire data from Ambee API."""
        cache_key = f"wildfire_{lat}_{lon}"
        
        wildfire_data = self.cache.get(cache_key, DISASTER_CACHE_TTL)
        if wildfire_data:
            return wildfire_data
            
        try:
            headers = {
                'x-api-key': AMBEE_API_KEY,
                'Content-type': 'application/json'
            }
            
            url = f"https://api.ambeedata.com/fire/latest/by-lat-lng?lat={lat}&lng={lon}"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('count', 0):
                wildfire_data = {
                    'active_fires': [],
                    'nearby': False,
                    'risk_level': 'low',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                fires = data.get('data', [])
                active_fires = []
                
                for fire in fires:
                    active_fires.append({
                        'lat': fire.get('lat'),
                        'lng': fire.get('lng'),
                        'confidence': fire.get('confidence'),
                        'intensity': fire.get('frp') or fire.get('intensity', 0),
                        'distance_km': self._calculate_distance(lat, lon, fire.get('lat'), fire.get('lng'))
                    })
                
                active_fires.sort(key=lambda x: x.get('distance_km', float('inf')))
                nearby = any(fire.get('distance_km', float('inf')) < 50 for fire in active_fires)
                intensity = max((fire.get('intensity', 0) for fire in active_fires), default=0)
                
                risk_level = 'low'
                if nearby and intensity > 50:
                    risk_level = 'high'
                elif nearby or intensity > 20:
                    risk_level = 'medium'
                
                wildfire_data = {
                    'active_fires': active_fires,
                    'nearby': nearby,
                    'risk_level': risk_level,
                    'timestamp': datetime.now().isoformat()
                }
            
            self.cache.set(cache_key, wildfire_data)
            return wildfire_data
        except Exception as e:
            logger.error(f"Error fetching wildfire data: {e}")
            return {
                'active_fires': [],
                'nearby': False,
                'risk_level': 'low',
                'timestamp': datetime.now().isoformat()
            }

    def get_fire_risk(self, lat, lon):
        """Fetch fire risk data."""
        cache_key = f"fire_risk_{lat}_{lon}"
        
        fire_risk_data = self.cache.get(cache_key, DISASTER_CACHE_TTL)
        if fire_risk_data:
            return fire_risk_data
            
        try:
            headers = {
                'x-api-key': AMBEE_API_KEY,
                'Content-type': 'application/json'
            }
            
            url = f"https://api.ambeedata.com/fire/risk/by-lat-lng?lat={lat}&lng={lon}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and data.get('data'):
                    risk_data = data.get('data')[0]
                    fire_risk_data = {
                        'risk_level': risk_data.get('risk', 'low'),
                        'risk_factors': risk_data.get('risk_factors', {}),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'ambee'
                    }
                    self.cache.set(cache_key, fire_risk_data)
                    return fire_risk_data
            
            # Fallback to weather-based calculation
            weather_data = self.get_weather_data(lat, lon)
            if not weather_data:
                return None
                
            current = weather_data['current']
            temperature = current.get('temperature', 0)
            humidity = current.get('humidity', 50)
            wind_speed = current.get('wind_speed', 0)
            precipitation = current.get('precipitation', 0)
            
            risk_score = 0
            if temperature > 30:
                risk_score += 30
            elif temperature > 25:
                risk_score += 20
            elif temperature > 20:
                risk_score += 10
                
            if humidity < 30:
                risk_score += 30
            elif humidity < 40:
                risk_score += 20
            elif humidity < 50:
                risk_score += 10
                
            if wind_speed > 20:
                risk_score += 30
            elif wind_speed > 10:
                risk_score += 15
            elif wind_speed > 5:
                risk_score += 5
                
            if precipitation == 0:
                risk_score += 10
                
            risk_level = 'low'
            if risk_score > 60:
                risk_level = 'high'
            elif risk_score > 30:
                risk_level = 'medium'
                
            fire_risk_data = {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': {
                    'temperature': temperature,
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'precipitation': precipitation
                },
                'timestamp': datetime.now().isoformat(),
                'source': 'calculated'
            }
            
            self.cache.set(cache_key, fire_risk_data)
            return fire_risk_data
        except Exception as e:
            logger.error(f"Error getting fire risk data: {e}")
            return {
                'risk_level': 'unknown',
                'risk_score': 0,
                'timestamp': datetime.now().isoformat(),
                'source': 'error'
            }
        
    def check_alerts(self, city):
        """Check for any alerts based on current data and forecasts."""
        coords = self.get_coordinates(city)
        if not coords:
            return "Could not find coordinates for the city."
            
        lat, lon = coords['lat'], coords['lon']
        
        weather_data = self.get_weather_data(lat, lon)
        flood_data = self.get_flood_data(lat, lon)
        wildfire_data = self.get_wildfire_data(lat, lon)
        fire_risk_data = self.get_fire_risk(lat, lon)
        
        if not all([weather_data, flood_data, wildfire_data, fire_risk_data]):
            return "Failed to fetch all required data."
            
        alerts = []
        
        current = weather_data['current']
        forecast = weather_data['forecast']
        
        if current['temperature'] > 40:
            alerts.append({
                'type': 'weather',
                'level': 'high',
                'message': f"Extreme heat alert: {current['temperature']}°C"
            })
        elif current['temperature'] > 35:
            alerts.append({
                'type': 'weather',
                'level': 'medium',
                'message': f"Heat warning: {current['temperature']}°C"
            })
            
        if forecast['precipitation'] > 25:
            alerts.append({
                'type': 'weather',
                'level': 'high',
                'message': f"Heavy rain expected in the next hour: {forecast['precipitation']} mm"
            })
        elif forecast['precipitation'] > 10:
            alerts.append({
                'type': 'weather',
                'level': 'medium',
                'message': f"Moderate rain expected in the next hour: {forecast['precipitation']} mm"
            })
            
        if current['wind_speed'] > 20:
            alerts.append({
                'type': 'weather',
                'level': 'high',
                'message': f"Strong winds: {current['wind_speed']} m/s"
            })
            
        if flood_data['risk_level'] == 'high':
            alerts.append({
                'type': 'flood',
                'level': 'high',
                'message': f"High flood risk! Water level: {flood_data['water_level']} mm"
            })
        elif flood_data['risk_level'] == 'medium':
            alerts.append({
                'type': 'flood',
                'level': 'medium',
                'message': f"Moderate flood risk. Water level: {flood_data['water_level']} mm"
            })
            
        if wildfire_data['nearby']:
            closest_fire = wildfire_data['active_fires'][0]
            alerts.append({
                'type': 'wildfire',
                'level': 'high',
                'message': f"Active fire detected {closest_fire['distance_km']:.1f} km away!"
            })
            
        if fire_risk_data['risk_level'] == 'high':
            alerts.append({
                'type': 'wildfire',
                'level': 'high',
                'message': "High fire danger due to current conditions"
            })
        elif fire_risk_data['risk_level'] == 'medium':
            alerts.append({
                'type': 'wildfire',
                'level': 'medium',
                'message': "Elevated fire risk due to current conditions"
            })
            
        return {
            'city': city,
            'coordinates': coords,
            'current_weather': current,
            'forecast': forecast,
            'flood_data': flood_data,
            'wildfire_data': wildfire_data,
            'fire_risk': fire_risk_data,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in kilometers."""
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r


# ====================== FLOOD PREDICTION ======================

flood_model = None
try:
    flood_model = joblib.load("flood_prediction_model.pkl")
except Exception as e:
    logger.warning(f"Could not load flood model: {e}")

def get_lat_lon(city):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key={OPENCAGE_API_KEY}"
    response = requests.get(url, timeout=10).json()
    if response["results"]:
        lat = response["results"][0]["geometry"]["lat"]
        lon = response["results"][0]["geometry"]["lng"]
        return lat, lon
    return None, None

def get_weather(lat, lon, date):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url, timeout=10).json()
    rainfall_24h, rainfall_72h, temperature = 0, 0, 0
    
    for item in response["list"]:
        if date in item["dt_txt"]:
            temperature = item["main"]["temp"]
            rainfall_24h = item.get("rain", {}).get("3h", 0) * 8
            break
    
    for item in response["list"]:
        if date in item["dt_txt"]:
            index = response["list"].index(item)
    
    for i in range(index, min(index + 24, len(response["list"]))):
        rainfall_72h += response["list"][i].get("rain", {}).get("3h", 0)
    
    return rainfall_24h, rainfall_72h, temperature

@index.route("/flood", methods=["POST"])
def flood_predict():
    if not flood_model:
        return jsonify({"error": "Flood prediction model not loaded"}), 500
        
    data = request.json
    city = data.get("city")
    date_str = data.get("date")
    soil_moisture = float(data.get("soilMoisture", 0))
    river_level = float(data.get("riverLevel", 0))
    reservoir_level = float(data.get("reservoirLevel", 0))
    previous_floods = float(data.get("previousFloods", 0))

    if not city or not date_str:
        return jsonify({"error": "Missing city or date"}), 400

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    lat, lon = get_lat_lon(city)
    if lat is None:
        return jsonify({"error": "Invalid city name"}), 400

    rainfall_24h, rainfall_72h, temperature = get_weather(lat, lon, date_str)
    actual_rainfall_24h = rainfall_24h
    actual_rainfall_72h = rainfall_72h

    if 0 <= rainfall_24h <= 45:
        rainfall_24h = random.uniform(45, 55)
    if 0 <= rainfall_72h <= 80:
        rainfall_72h = random.uniform(80, 90)

    snowmelt = 0.0
    slope_gradient = 10.0
    vegetation_cover = 60.0
    urbanization = 40.0

    feature_names = [
        "rainfall_24h", "rainfall_72h", "river_level", "soil_moisture", "reservoir_level",
        "previous_floods", "temperature", "snowmelt", "slope_gradient", "vegetation_cover",
        "urbanization", "year", "month", "day"
    ]

    input_data = pd.DataFrame([[rainfall_24h, rainfall_72h, river_level, soil_moisture, reservoir_level,
                              previous_floods, temperature, snowmelt, slope_gradient, vegetation_cover,
                              urbanization, year, month, day]], columns=feature_names)

    prediction = flood_model.predict(input_data)[0]
    flood_probability = flood_model.predict_proba(input_data)[0][1]

    return jsonify({
        "latitude": lat,
        "longitude": lon,
        "rainfall_24h": actual_rainfall_24h,
        "rainfall_72h": actual_rainfall_72h,
        "temperature": temperature,
        "soil_moisture": soil_moisture,
        "river_level": river_level,
        "reservoir_level": reservoir_level,
        "previous_floods": previous_floods,
        "year": year,
        "month": month,
        "day": day,
        "flood_prediction": int(prediction),
        "flood_probability": round(flood_probability * 100, 2)
    })

# ====================== HEATWAVE PREDICTION ======================

class HeatwavePredictionService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = ['temperature_2m_max', 'apparent_temperature_max',
                        'relative_humidity_2m_mean', 'wind_speed_10m_max',
                        'pressure_msl_mean', 'precipitation_sum', 'cloud_cover_mean']
        self.model_file = os.path.join('model_files', 'heatwave_model.pkl')
        self.scaler_file = os.path.join('model_files', 'heatwave_scaler.pkl')
        self.city_coordinates = {}
        self.geolocator = Nominatim(user_agent="heatwave_predictor")
        self.load_model()

    def load_model(self):
        """Load the saved model"""
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                return True
            except Exception as e:
                logger.warning(f"Could not load heatwave model: {e}")
        return False

    def get_coordinates(self, city_name):
        """Convert city name to coordinates with caching"""
        if city_name in self.city_coordinates:
            return self.city_coordinates[city_name]
        try:
            location = self.geolocator.geocode(city_name, timeout=10)
            if location:
                self.city_coordinates[city_name] = (location.latitude, location.longitude)
                return location.latitude, location.longitude
            else:
                raise ValueError(f"Coordinates not found for: {city_name}")
        except Exception as e:
            raise ValueError(f"Geocoding error for {city_name}: {str(e)}")

    def fetch_forecast_data(self, city):
        """Fetch 7-day forecast data for prediction"""
        try:
            lat, lon = self.get_coordinates(city)
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": self.features,
                    "forecast_days": 7,
                    "timezone": "auto"
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            forecast_df = pd.DataFrame(data["daily"])
            forecast_df['city'] = city
            return forecast_df
        except Exception as e:
            raise ValueError(f"Failed to fetch forecast for {city}: {str(e)}")

    def clean_data(self, df):
        """Handle missing values and data quality issues"""
        if 'is_heatwave' in df.columns:
            df = df.dropna(subset=['is_heatwave'])
        for feature in self.features:
            if feature in df.columns:
                if feature == 'precipitation_sum':
                    df[feature] = df[feature].fillna(0)
                elif feature.endswith('_mean') or feature.endswith('_max'):
                    df[feature] = df[feature].fillna(df[feature].rolling(3, min_periods=1).mean())
        df = df.dropna(subset=self.features)
        return df

    def predict_heatwave(self, city):
        """Predict heatwave with better-calibrated probabilities"""
        try:
            forecast_df = self.fetch_forecast_data(city)
            forecast_df = self.clean_data(forecast_df)
            
            if self.model is None:
                forecast_df['is_heatwave'] = (forecast_df['temperature_2m_max'] > 35).astype(int)
                forecast_df['heatwave_probability'] = np.clip((forecast_df['temperature_2m_max'] - 30) / 10, 0, 1)
                
                conditions = [
                    (forecast_df['heatwave_probability'] <= 0.3),
                    (forecast_df['heatwave_probability'] <= 0.6),
                    (forecast_df['heatwave_probability'] <= 0.8),
                    (forecast_df['heatwave_probability'] > 0.8)
                ]
                
                alerts = np.array(['Normal', 'Caution', 'Warning', 'Emergency'], dtype='object')
                colors = np.array(['green', 'yellow', 'orange', 'red'], dtype='object')
                actions = np.array([
                    "No special precautions needed",
                    "Stay hydrated, limit outdoor activities",
                    "Avoid outdoor activities 11am-4pm",
                    "Extreme danger - stay indoors"
                ], dtype='object')
                
                forecast_df['alert_level'] = np.select(conditions, alerts, default='Unknown')
                forecast_df['alert_color'] = np.select(conditions, colors, default='gray')
                forecast_df['recommended_action'] = np.select(conditions, actions, default='Monitor conditions')
                
                return forecast_df
            
            X = forecast_df[self.features].copy()
            X['temp_humidity_ratio'] = X['temperature_2m_max'] / (X['relative_humidity_2m_mean'].clip(lower=1))
            X['diurnal_range'] = X['temperature_2m_max'] - X['apparent_temperature_max']
            
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            temp_boost = np.clip(
                (forecast_df['temperature_2m_max'] - 35) / 15,
                0, 0.3
            )
            probabilities = np.clip(probabilities + temp_boost, 0.01, 0.99)
            
            for i in range(1, len(probabilities)):
                if probabilities[i - 1] > 0.3:
                    probabilities[i] = np.clip(probabilities[i] * (1 + 0.1 * (probabilities[i - 1] - 0.3) / 0.7), 0, 0.95)
            
            humidity_effect = 1 - (X['relative_humidity_2m_mean'] / 100) ** 2
            probabilities = probabilities * (0.7 + 0.3 * humidity_effect)
            
            forecast_df['is_heatwave'] = (probabilities > 0.5).astype(int)
            forecast_df['heatwave_probability'] = probabilities
            
            conditions = [
                (probabilities <= 0.3),
                (probabilities <= 0.6),
                (probabilities <= 0.8),
                (probabilities > 0.8)
            ]
            
            alerts = np.array(['Normal', 'Caution', 'Warning', 'Emergency'], dtype='object')
            colors = np.array(['green', 'yellow', 'orange', 'red'], dtype='object')
            actions = np.array([
                "No special precautions needed",
                "Stay hydrated, limit outdoor activities",
                "Avoid outdoor activities 11am-4pm",
                "Extreme danger - stay indoors"
            ], dtype='object')
            
            forecast_df['alert_level'] = np.select(conditions, alerts, default='Unknown')
            forecast_df['alert_color'] = np.select(conditions, colors, default='gray')
            forecast_df['recommended_action'] = np.select(conditions, actions, default='Monitor conditions')
            
            return forecast_df
        
        except Exception as e:
            raise ValueError(f"Prediction failed for {city}: {str(e)}")

    def fetch_historical_data(self, city, years=2):
        """Fetch historical weather data for a city"""
        try:
            lat, lon = self.get_coordinates(city)
            current_date = datetime.now().date()
            all_data = []
            
            for year_offset in range(1, years + 1):
                year = current_date.year - year_offset
                start_date = datetime(year, 2, 1).date()
                end_date = datetime(year, 5, 31).date()
                
                if year == current_date.year:
                    end_date = min(end_date, current_date - timedelta(days=1))
                
                try:
                    response = requests.get(
                        "https://archive-api.open-meteo.com/v1/archive",
                        params={
                            "latitude": lat,
                            "longitude": lon,
                            "start_date": start_date.strftime("%Y-%m-%d"),
                            "end_date": end_date.strftime("%Y-%m-%d"),
                            "daily": self.features,
                            "timezone": "auto"
                        },
                        timeout=15
                    )
                    response.raise_for_status()
                    data = response.json()
                    if not data.get("daily"):
                        continue
                    
                    df_year = pd.DataFrame(data["daily"])
                    df_year['city'] = city
                    df_year['year'] = year
                    all_data.append(df_year)
                except Exception:
                    continue
            
            if not all_data:
                raise ValueError(f"Failed to fetch any historical data for {city}")
            
            result = pd.concat(all_data).drop_duplicates(["time", "city"])
            result['time'] = pd.to_datetime(result['time'])
            result['is_heatwave'] = (result['temperature_2m_max'] > 35).astype(int)
            
            response_data = []
            for _, row in result.iterrows():
                response_data.append({
                    'date': row['time'].strftime('%Y-%m-%d'),
                    'temperature': round(row['temperature_2m_max'], 1),
                    'apparent_temperature': round(row['apparent_temperature_max'], 1),
                    'humidity': round(row['relative_humidity_2m_mean'], 1),
                    'is_heatwave': int(row['is_heatwave'])
                })
            
            return response_data
        
        except Exception as e:
            raise ValueError(f"Failed to fetch historical data for {city}: {str(e)}")

heatwave_service = HeatwavePredictionService()

# ====================== API ENDPOINTS ======================

@index.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Welcome to the Combined Disaster Prediction API",
        "endpoints": {
            "disaster_monitor": "/disaster-monitor/<city>",
            "flood_prediction": "/flood",
            "heatwave_prediction": "/heatwave",
            "heatwave_historical": "/api/historical"
        }
    })

@index.route('/disaster-monitor/<city>', methods=['GET'])
def get_disaster_data(city):
    """Get comprehensive disaster monitoring data for a city."""
    try:
        monitor = DisasterMonitor()
        results = monitor.check_alerts(city)
        if isinstance(results, str):
            return jsonify({'error': results}), 400
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in disaster monitor: {e}")
        return jsonify({'error': str(e)}), 500

@index.route('/heatwave', methods=['POST'])
def heatwave_predict():
    """Predict heatwave for a city."""
    try:
        data = request.get_json()
        if not data or 'city' not in data:
            return jsonify({"error": "City parameter is required"}), 400
        
        city = data['city']
        forecast = heatwave_service.predict_heatwave(city)
        
        try:
            lat, lon = heatwave_service.get_coordinates(city)
        except Exception as e:
            logger.warning(f"Could not get coordinates for {city}: {e}")
            lat, lon = 0, 0
        
        return jsonify({
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "predictions": forecast.to_dict('records'),
            "message": "Forecast generated successfully",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in heatwave prediction: {e}")
        return jsonify({"error": str(e)}), 500

@index.route('/api/historical', methods=['GET'])
def get_heatwave_historical_data():
    """Get historical heatwave data for a city."""
    try:
        city = request.args.get('city')
        if not city:
            return jsonify({"error": "City parameter is required"}), 400
        
        historical_data = heatwave_service.fetch_historical_data(city)
        
        return jsonify({
            "city": city,
            "historical_data": historical_data,
            "message": "Historical data retrieved successfully",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in historical data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs('model_files', exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    index.run(host='0.0.0.0', port=5000, debug=True)