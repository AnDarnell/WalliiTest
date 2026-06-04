import requests
r = requests.get("https://api.hearthstonejson.com/v1/latest/enUS/cards.json")
cards = r.json()

spell = next(c for c in cards if c.get("name") == "Enchanted Lasso")
import json
print(json.dumps(spell, indent=2))