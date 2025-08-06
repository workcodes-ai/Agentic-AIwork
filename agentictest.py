import requests

class WeatherAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.weatherapi.com/v1/current.json"

    def get_weather(self, location):
        params = {
            "key": self.api_key,
            "q": location,
            "aqi": "no"
        }

        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            current = data['current']
            location = data['location']
            return {
                "location": f"{location['name']}, {location['country']}",
                "temperature_c": current['temp_c'],
                "condition": current['condition']['text'],
                "feelslike_c": current['feelslike_c'],
                "wind_kph": current['wind_kph']
            }
        else:
            return {"error": f"Failed to retrieve data: {response.status_code}"}

# Example usage
if __name__ == "__main__":
    API_KEY = "your_weatherapi_key_here"
    agent = WeatherAgent(API_KEY)
    location = input("Enter a location: ")
    weather = agent.get_weather(location)

    if "error" in weather:
        print(weather["error"])
    else:
        print(f"Weather in {weather['location']}:")
        print(f"Temperature: {weather['temperature_c']}°C")
        print(f"Condition: {weather['condition']}")
        print(f"Feels like: {weather['feelslike_c']}°C")
        print(f"Wind: {weather['wind_kph']} kph")
