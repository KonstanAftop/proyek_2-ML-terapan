import requests
import zipfile
import io
import os

url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

output_dir = "ml-100k"
os.makedirs(output_dir, exist_ok=True)

response = requests.get(url)
if response.status_code == 200:
    print("Download successful. Extracting...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(output_dir)
    print(f"Dataset extracted to: {output_dir}/")
else:
    print(f"Failed to download the dataset. Status code: {response.status_code}")