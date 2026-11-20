import os
import requests

MODEL_DIR = os.path.join("models", "ms-marco-MiniLM-L-6-v2")
BASE_URL = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main"

FILES = [
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "model.safetensors",
]

def download_reranker():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename in FILES:
        dest_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"[✓] Already downloaded: {filename}")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {filename}...")
        resp = requests.get(url, stream=True, timeout=30, allow_redirects=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded / total * 100)
                        print(f"  -> {filename}: {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({pct}%)", end="\r")
        print(f"\n[+] Saved {filename}")

    print("\n[+] Cross-encoder model successfully saved to:", MODEL_DIR)

if __name__ == "__main__":
    download_reranker()
