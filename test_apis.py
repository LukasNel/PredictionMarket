import requests
import json
BASE_URL = "https://gamma-api.polymarket.com"

with open("polymarket_markets.json", "w") as f:
    json.dump(requests.get(f"{BASE_URL}/markets").json(), f, indent=4)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
with open("kalshi_markets.json", "w") as f:
    json.dump(requests.get(f"{BASE_URL}/markets").json(), f, indent=4)