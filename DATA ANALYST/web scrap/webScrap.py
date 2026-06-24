# Step 1: Import required libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 2: Define the target URL
url = "https://www.zomato.com/bangalore/restaurants"  # Example city page

# Step 3: Add headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0 Safari/537.36"
}

# Step 4: Send HTTP request with headers
response = requests.get(url, headers=headers)
if response.status_code != 200:
    print("Failed to retrieve page:", response.status_code)
    exit()

# Step 5: Parse HTML content
soup = BeautifulSoup(response.text, "html.parser")

# Step 6: Extract restaurant data (adjust selectors after inspecting HTML)
restaurants = []
for item in soup.find_all("a", class_="sc-dpiBDa"):  # Example selector
    name = item.get_text(strip=True)
    restaurants.append({"Name": name})

# Step 7: Convert to DataFrame
df = pd.DataFrame(restaurants)

# Step 8: Save to CSV
df.to_csv("restaurants.csv", index=False)

print("Scraping complete! Data saved to restaurants.csv")
