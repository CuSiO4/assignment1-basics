# download_data.py
import urllib.request
import gzip
import os

os.makedirs("data", exist_ok=True)
os.chdir("data")

urls = {
    "TinyStoriesV2-GPT4-train.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "owt_train.txt.gz": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
    "owt_valid.txt.gz": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
}

for name, url in urls.items():
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, name)

# Decompress .gz files
for gz_name in ["owt_train.txt.gz", "owt_valid.txt.gz"]:
    with gzip.open(gz_name, "rb") as f_in, open(gz_name[:-3], "wb") as f_out:
        f_out.write(f_in.read())