import requests, re, json

r = requests.get(
    "https://hearthstone.blizzard.com/en-us/api/blog/articleList",
    params={"page": 1, "pageSize": 20, "category": "patchnotes", "locale": "en_US"},
    headers={"User-Agent": "Mozilla/5.0"}
)
articles = r.json()
for a in articles:
    title = a.get("title", "")
    slug = a.get("slug", "")
    if re.match(r'^\d+\.\d+', title):
        print(title, "|", f"https://hearthstone.blizzard.com/en-us/news/{a['id']}/{slug}")