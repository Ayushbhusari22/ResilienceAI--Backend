import os
import sys
import json
import time
import logging
import requests
import argparse
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
from flask import Flask, jsonify
from flask_cors import CORS  # Add this import
app = Flask(__name__)
CORS(app)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("disaster_monitor.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# API Keys
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY")
AMBEE_API_KEY = os.environ.get("AMBEE_API_KEY")
# Cache settings
CACHE_DIR = "cache"
WEATHER_CACHE_TTL = 5 * 60  # 5 minutes for weather data
DISASTER_CACHE_TTL = 10 * 60  # 10 minutes for flood and wildfire data

# Rich console for prettier output
console = Console()

PORT = os.environ.get('PORT', 5000)
PRODUCTION_MODE = os.environ.get('PRODUCTION_MODE', False)


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
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            timestamp = cache_data.get('timestamp', 0)
            if time.time() - timestamp > ttl:
                return None  # Cache expired
                
            return cache_data.get('data')
            
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def set(self, key, data):
        """Store data in cache with current timestamp."""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        try:
            cache_data = {
                'data': data,
                'timestamp': time.time()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")


class DisasterMonitor:
    """Main class to fetch and process disaster-related data."""

    def __init__(self):
        self.cache = Cache()
        
    def get_coordinates(self, city):
        """Get latitude and longitude for a given city using WeatherAPI as primary source."""
        cache_key = f"geocode_{city.lower().replace(' ', '_')}"
        
        # Check cache first
        coords = self.cache.get(cache_key, 86400)  # Cache for 24 hours
        if coords:
            return coords
            
        try:
            # Primary: Use WeatherAPI geocoding
            url = f"http://api.weatherapi.com/v1/search.json?key={WEATHERAPI_KEY}&q={city}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if data and isinstance(data, list) and len(data) > 0:
                # Select the first result (most relevant) or implement logic to find best match
                location = data[0]
                coords = {
                    'lat': float(location.get('lat')),
                    'lon': float(location.get('lon')),
                    'name': location.get('name'),
                    'country': location.get('country'),
                    'admin1': location.get('region'),
                    'source': 'weatherapi'
                }
                
                # Additional validation
                if not self.validate_coordinates(coords['lat'], coords['lon']):
                    raise ValueError("Invalid coordinates from WeatherAPI")
                    
                self.cache.set(cache_key, coords)
                return coords
                
        except Exception as e:
            logger.error(f"Error getting coordinates from WeatherAPI: {e}")
            # Fallback to Open-Meteo if WeatherAPI fails
            try:
                url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
                response = requests.get(url)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('results'):
                    location = data['results'][0]
                    coords = {
                        'lat': location.get('latitude'),
                        'lon': location.get('longitude'),
                        'name': location.get('name'),
                        'country': location.get('country'),
                        'admin1': location.get('admin1'),
                        'source': 'open-meteo'
                    }
                    
                    if not self.validate_coordinates(coords['lat'], coords['lon']):
                        raise ValueError("Invalid coordinates from Open-Meteo")
                        
                    self.cache.set(cache_key, coords)
                    return coords
                    
            except Exception as fallback_e:
                logger.error(f"Error getting coordinates from Open-Meteo: {fallback_e}")
                
        logger.error(f"Could not find coordinates for {city} using any geocoding service")
        return None
    
    def validate_coordinates(self, lat, lon):
        """Check if coordinates are within Earth's bounds."""
        return -90 <= lat <= 90 and -180 <= lon <= 180
        
    def get_weather_data(self, lat, lon):
        """Fetch current weather and forecast data with improved precipitation detection."""
        cache_key = f"weather_{lat}_{lon}"
        
        # Check cache first
        weather_data = self.cache.get(cache_key, WEATHER_CACHE_TTL)
        if weather_data:
            return weather_data
            
        try:
            # First try WeatherAPI for more accurate precipitation data
            url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={lat},{lon}&days=1&aqi=no&alerts=no"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            current = data.get('current', {})
            forecast_hour = data.get('forecast', {}).get('forecastday', [{}])[0].get('hour', [{}])[1]  # Next hour
            
            weather_data = {
                'current': {
                    'temperature': current.get('temp_c'),
                    'humidity': current.get('humidity'),
                    'wind_speed': current.get('wind_kph') / 3.6,  # Convert to m/s
                    'pressure': current.get('pressure_mb'),
                    'precipitation': current.get('precip_mm', 0),
                    'conditions': current.get('condition', {}).get('text', 'Unknown'),
                    'description': current.get('condition', {}).get('text', 'Unknown'),
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
                    'description': forecast_hour.get('condition', {}).get('text', 'Unknown'),
                    'timestamp': forecast_hour.get('time', (datetime.now() + timedelta(hours=1)).isoformat())
                }
            }
            
            self.cache.set(cache_key, weather_data)
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data from WeatherAPI: {e}")
            # Fallback to Open-Meteo if WeatherAPI fails
            try:
                url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain,pressure_msl,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,rain,pressure_msl,wind_speed_10m&timezone=auto"
                response = requests.get(url)
                response.raise_for_status()
                
                data = response.json()
                current = data.get('current', {})
                hourly = data.get('hourly', {})
                
                next_hour = {
                    'temperature': hourly.get('temperature_2m', [None])[1],
                    'humidity': hourly.get('relative_humidity_2m', [None])[1],
                    'precipitation': max(hourly.get('precipitation', [0])[1], hourly.get('rain', [0])[1]),
                    'probability': hourly.get('precipitation_probability', [0])[1],
                    'wind_speed': hourly.get('wind_speed_10m', [0])[1],
                    'pressure': hourly.get('pressure_msl', [0])[1]
                }
                
                weather_data = {
                    'current': {
                        'temperature': current.get('temperature_2m'),
                        'humidity': current.get('relative_humidity_2m'),
                        'wind_speed': current.get('wind_speed_10m'),
                        'pressure': current.get('pressure_msl'),
                        'precipitation': max(current.get('precipitation', 0), current.get('rain', 0)),
                        'conditions': self._get_weather_condition(current),
                        'description': self._get_weather_description(current),
                        'timestamp': datetime.now().isoformat()
                    },
                    'forecast': {
                        'temperature': next_hour.get('temperature'),
                        'humidity': next_hour.get('humidity'),
                        'wind_speed': next_hour.get('wind_speed'),
                        'pressure': next_hour.get('pressure'),
                        'precipitation': next_hour.get('precipitation', 0),
                        'probability': next_hour.get('probability', 0),
                        'conditions': self._get_weather_condition(next_hour),
                        'description': self._get_weather_description(next_hour),
                        'timestamp': (datetime.now() + timedelta(hours=1)).isoformat()
                    }
                }
                
                self.cache.set(cache_key, weather_data)
                return weather_data
                
            except Exception as fallback_e:
                logger.error(f"Error fetching weather data from Open-Meteo: {fallback_e}")
                return None
        
    def cross_verify_weather(self, lat, lon):
        """Verify weather data with a secondary API."""
        try:
            # Example: WeatherAPI (requires a free API key)
            url = f"http://api.weatherapi.com/v1/current.json?key=24dda02be42748e1a93142205250304&q={lat},{lon}"
            response = requests.get(url).json()
            return {
                'temp': response['current']['temp_c'],
                'humidity': response['current']['humidity'],
                'wind': response['current']['wind_kph'] / 3.6,  # Convert to m/s
            }
        except Exception as e:
            logger.error(f"WeatherAPI verification failed: {e}")
            return None

    def _get_weather_condition(self, data):
        """More accurate weather condition detection."""
        precip = data.get('precipitation', 0)
        wind_speed = data.get('wind_speed', 0)
        
        if precip > 7.5:
            return 'Heavy Rain'
        elif precip > 2.5:
            return 'Moderate Rain'
        elif precip > 0.1:
            return 'Light Rain'
        elif wind_speed > 15:
            return 'Windy'
        elif wind_speed > 8:
            return 'Breezy'
        return 'Clear'
    
    def _get_weather_description(self, data):
        """Generate weather description based on available data."""
        desc = []
        if data.get('rain', 0) > 5:
            desc.append(f"Heavy rain ({data.get('rain')}mm)")
        elif data.get('rain', 0) > 0:
            desc.append(f"Light rain ({data.get('rain')}mm)")
        
        if data.get('wind_speed', 0) > 10:
            desc.append(f"Windy ({data.get('wind_speed')} m/s)")
        
        return ", ".join(desc) if desc else "Clear skies"
        
    def get_flood_data(self, lat, lon):
        """Enhanced flood risk assessment with better precipitation handling."""
        cache_key = f"flood_{lat}_{lon}"
        
        # Check cache first
        flood_data = self.cache.get(cache_key, DISASTER_CACHE_TTL)
        if flood_data:
            return flood_data
            
        try:
            weather_data = self.get_weather_data(lat, lon)
            if not weather_data:
                return None
                
            current = weather_data['current']
            forecast = weather_data['forecast']
            
            # Get precipitation from multiple sources if available
            current_precip = current.get('precipitation', 0)
            forecast_precip = forecast.get('precipitation', 0)
            forecast_prob = forecast.get('probability', 0)
            
            # Consider terrain elevation if available (hilly areas like Ooty are more prone to flooding)
            elevation_factor = 1.2 if self._is_hilly_terrain(lat, lon) else 1.0
            
            # Enhanced flood risk calculation
            water_level = max(current_precip, forecast_precip) * elevation_factor
            risk_level = 'low'
            
            # Adjusted thresholds for different terrains
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
        """Simple terrain detection (could be enhanced with elevation API)."""
        # Known hilly regions in India
        hilly_regions = [
            (11.41, 76.70, 50),  # Ooty approximate center and radius
            (30.37, 78.96, 100),  # Dehradun/Mussoorie
            (15.35, 76.16, 80)    # Coorg
        ]
        
        for region_lat, region_lon, radius in hilly_regions:
            if self._calculate_distance(lat, lon, region_lat, region_lon) < radius:
                return True
        return False
    
    def get_wildfire_data(self, lat, lon):
        """Fetch wildfire data from Ambee API."""
        cache_key = f"wildfire_{lat}_{lon}"
        
        # Check cache first
        wildfire_data = self.cache.get(cache_key, DISASTER_CACHE_TTL)
        if wildfire_data:
            return wildfire_data
            
        try:
            # Using Ambee fire API
            headers = {
                'x-api-key': AMBEE_API_KEY,
                'Content-type': 'application/json'
            }
            
            url = f"https://api.ambeedata.com/fire/latest/by-lat-lng?lat={lat}&lng={lon}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('count', 0):
                # No fires detected
                wildfire_data = {
                    'active_fires': [],
                    'nearby': False,
                    'risk_level': 'low',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                fires = data.get('data', [])
                
                # Process fire data
                active_fires = []
                for fire in fires:
                    active_fires.append({
                        'lat': fire.get('lat'),
                        'lng': fire.get('lng'),
                        'confidence': fire.get('confidence'),
                        'intensity': fire.get('frp') or fire.get('intensity', 0),
                        'distance_km': self._calculate_distance(lat, lon, fire.get('lat'), fire.get('lng'))
                    })
                
                # Sort by distance
                active_fires.sort(key=lambda x: x.get('distance_km', float('inf')))
                
                # Determine risk level
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
            # Return a default response on error
            return {
                'active_fires': [],
                'nearby': False,
                'risk_level': 'low',
                'timestamp': datetime.now().isoformat()
            }

    def get_fire_risk(self, lat, lon):
        """Fetch fire risk data from Ambee API or calculate based on weather."""
        cache_key = f"fire_risk_{lat}_{lon}"
        
        # Check cache first
        fire_risk_data = self.cache.get(cache_key, DISASTER_CACHE_TTL)
        if fire_risk_data:
            return fire_risk_data
            
        try:
            # First try Ambee API
            headers = {
                'x-api-key': AMBEE_API_KEY,
                'Content-type': 'application/json'
            }
            
            url = f"https://api.ambeedata.com/fire/risk/by-lat-lng?lat={lat}&lng={lon}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data and data.get('data'):
                    # Use Ambee's risk data
                    risk_data = data.get('data')[0]
                    fire_risk_data = {
                        'risk_level': risk_data.get('risk', 'low'),
                        'risk_factors': risk_data.get('risk_factors', {}),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'ambee'
                    }
                    self.cache.set(cache_key, fire_risk_data)
                    return fire_risk_data
            
            # Fallback to calculating based on weather
            weather_data = self.get_weather_data(lat, lon)
            if not weather_data:
                return None
                
            current = weather_data['current']
            
            # Factors that increase fire risk:
            # - High temperature
            # - Low humidity
            # - High wind speed
            # - Low precipitation
            
            temperature = current.get('temperature', 0)
            humidity = current.get('humidity', 50)
            wind_speed = current.get('wind_speed', 0)
            precipitation = current.get('precipitation', 0)
            
            # Simple algorithm for fire risk
            risk_score = 0
            
            # Temperature factor (0-40°C scale)
            if temperature > 30:
                risk_score += 30
            elif temperature > 25:
                risk_score += 20
            elif temperature > 20:
                risk_score += 10
                
            # Humidity factor (inverse relationship)
            if humidity < 30:
                risk_score += 30
            elif humidity < 40:
                risk_score += 20
            elif humidity < 50:
                risk_score += 10
                
            # Wind factor
            if wind_speed > 20:
                risk_score += 30
            elif wind_speed > 10:
                risk_score += 15
            elif wind_speed > 5:
                risk_score += 5
                
            # Precipitation factor (inverse)
            if precipitation == 0:
                risk_score += 10
                
            # Determine risk level based on score
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
        
    def calculate_fire_risk(self, temp, humidity, wind, precipitation):
        """Enhanced fire risk using Canadian FWI principles."""
        # Drought factor (0-10 scale)
        drought_factor = 10 if precipitation == 0 else max(0, 10 - precipitation * 2)
        
        # Wind impact (0-5 scale)
        wind_impact = min(5, wind / 5)
        
        # Humidity impact (0-5 scale)
        humidity_impact = 0 if humidity > 60 else (60 - humidity) / 12
        
        risk_score = (temp * 0.1) + drought_factor + wind_impact + humidity_impact
        
        if risk_score > 20:
            return "high"
        elif risk_score > 10:
            return "medium"
        return "low"
    
    def check_historical_deviation(self, lat, lon):
        """Compare current temp with 10-year average."""
        today = datetime.now()
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={today.year-10}-{today.month}-{today.day}&end_date={today.year-1}-{today.month}-{today.day}&daily=temperature_2m_mean"
        response = requests.get(url).json()
        historical_avg = sum(response['daily']['temperature_2m_mean']) / len(response['daily']['temperature_2m_mean'])
        current_temp = self.get_weather_data(lat, lon)['current']['temperature']
        return current_temp > historical_avg * 1.2  # 20% hotter than average
    
    def check_alerts(self, city):
        """Check for any alerts based on current data and forecasts."""
        coords = self.get_coordinates(city)
        if not coords:
            return "Could not find coordinates for the city."
            
        lat, lon = coords['lat'], coords['lon']
        
        # Get all data
        weather_data = self.get_weather_data(lat, lon)
        flood_data = self.get_flood_data(lat, lon)
        wildfire_data = self.get_wildfire_data(lat, lon)
        fire_risk_data = self.get_fire_risk(lat, lon)
        
        if not all([weather_data, flood_data, wildfire_data, fire_risk_data]):
            return "Failed to fetch all required data."
            
        alerts = []
        
        # Weather alerts
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
            
        # Flood alerts
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
            
        # Wildfire alerts
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

    def display_results(self, results):
        """Display the monitoring results in a nice format."""
        city = results.get('city', 'Unknown Location')
        coords = results.get('coordinates', {})
        weather = results.get('current_weather', {})
        forecast = results.get('forecast', {})
        flood = results.get('flood_data', {})
        wildfire = results.get('wildfire_data', {})
        fire_risk = results.get('fire_risk', {})
        alerts = results.get('alerts', [])
        
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Title
        console.print(f"[bold cyan]Disaster Monitor for {city}[/bold cyan]")
        console.print(f"Coordinates: {coords.get('lat', 'N/A')}, {coords.get('lon', 'N/A')}")
        console.print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print()
        
        # Current conditions
        weather_table = Table(title="Current Weather Conditions", box=box.ROUNDED)
        weather_table.add_column("Metric", style="cyan")
        weather_table.add_column("Current Value", style="green")
        weather_table.add_column("Next Hour", style="yellow")
        
        weather_table.add_row("Temperature", f"{weather.get('temperature', 'N/A')}°C", f"{forecast.get('temperature', 'N/A')}°C")
        weather_table.add_row("Conditions", weather.get('conditions', 'N/A'), forecast.get('conditions', 'N/A'))
        weather_table.add_row("Humidity", f"{weather.get('humidity', 'N/A')}%", f"{forecast.get('humidity', 'N/A')}%")
        weather_table.add_row("Wind Speed", f"{weather.get('wind_speed', 'N/A')} m/s", f"{forecast.get('wind_speed', 'N/A')} m/s")
        weather_table.add_row("Precipitation", f"{weather.get('precipitation', 'N/A')} mm", f"{forecast.get('precipitation', 'N/A')} mm")
        weather_table.add_row("Precipitation Probability", "Current", f"{forecast.get('probability', 'N/A')}%")
        
        console.print(weather_table)
        console.print()
        
        # Disaster Risks
        risk_table = Table(title="Disaster Risk Assessment", box=box.ROUNDED)
        risk_table.add_column("Disaster Type", style="cyan")
        risk_table.add_column("Risk Level", style="green")
        risk_table.add_column("Details", style="yellow")
        
        # Flood risk
        flood_risk = flood.get('risk_level', 'unknown')
        flood_color = {"high": "red", "medium": "yellow", "low": "green"}.get(flood_risk, "white")
        risk_table.add_row(
            "Flood", 
            f"[{flood_color}]{flood_risk.upper()}[/{flood_color}]",
            f"Water Level: {flood.get('water_level', 'N/A')} mm, Probability: {flood.get('probability', 'N/A')}%"
        )
        
        # Wildfire risk
        fire_level = fire_risk.get('risk_level', 'unknown')
        fire_color = {"high": "red", "medium": "yellow", "low": "green"}.get(fire_level, "white")
        risk_table.add_row(
            "Fire Risk", 
            f"[{fire_color}]{fire_level.upper()}[/{fire_color}]",
            f"Based on temperature, humidity, and wind conditions"
        )
        
        # Active fires
        active_fires = wildfire.get('active_fires', [])
        if active_fires:
            closest = active_fires[0]
            risk_table.add_row(
                "Active Fires", 
                f"[red]DETECTED[/red]",
                f"Closest fire: {closest.get('distance_km', 'N/A'):.1f} km away"
            )
        else:
            risk_table.add_row("Active Fires", "[green]NONE DETECTED[/green]", "No active fires in the vicinity")
        
        console.print(risk_table)
        console.print()
        
        # Alerts
        if alerts:
            alert_panel = Panel(
                "\n".join([f"[{'red' if a['level'] == 'high' else 'yellow'}]• {a['message']}[/]" for a in alerts]),
                title=f"[bold]ALERTS ({len(alerts)})[/bold]",
                border_style="red" if any(a['level'] == 'high' for a in alerts) else "yellow"
            )
            console.print(alert_panel)
        else:
            console.print("[green]No alerts at this time.[/green]")
            
        console.print()
        console.print("Press Ctrl+C to exit.")

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in kilometers."""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r


def main():
    """Main function to run the disaster monitoring script."""
    parser = argparse.ArgumentParser(description='Monitor disaster risks for a city')
    parser.add_argument('--city', '-c', type=str, help='City to monitor')
    parser.add_argument('--interval', '-i', type=int, default=300, help='Refresh interval in seconds (default: 300)')
    args = parser.parse_args()
    
    monitor = DisasterMonitor()
    
    # Get city name
    city = args.city
    if not city:
        console.print("[bold cyan]Enter the city name to monitor:[/bold cyan]")
        city = input("> ")
    
    try:
        while True:
            try:
                results = monitor.check_alerts(city)
                if isinstance(results, str):
                    console.print(f"[bold red]Error:[/bold red] {results}")
                    sys.exit(1)
                    
                monitor.display_results(results)
                
                # Wait for the specified interval
                time.sleep(args.interval)
                
            except KeyboardInterrupt:
                console.print("\n[bold cyan]Exiting Disaster Monitor...[/bold cyan]")
                sys.exit(0)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                console.print(f"[bold red]Error:[/bold red] {e}")
                time.sleep(10)  # Short delay before retrying
                
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Exiting Disaster Monitor...[/bold cyan]")
        sys.exit(0)


# if __name__ == "__main__":
#     main()

@app.route('/disaster-monitor/<city>', methods=['GET'])
def get_disaster_data(city):
    monitor = DisasterMonitor()
    results = monitor.check_alerts(city)
    if isinstance(results, str):
        return jsonify({'error': results}), 400
    return jsonify(results)

if __name__ == "__main__":
    os.makedirs('model_files', exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # For production on Render
    if PRODUCTION_MODE:
        app.run(host='0.0.0.0', port=int(PORT), debug=False)
    else:
        app.run(host='0.0.0.0', port=int(PORT), debug=True)