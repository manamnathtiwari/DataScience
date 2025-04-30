import requests

# GitHub Credentials (Replace with your actual values)
TOKEN = "64360d93179e7f9f746cb00a2909cfa5ba961f84"  # Replace with a valid GitHub PAT
REPO_OWNER = "ExperimentedSperm"  # Your GitHub username
REPO_NAME = "PythonFromScratch"  # Your repository name
FILE_PATH = "Day1.py"  # The specific file you want to fetch

# GitHub API URL to fetch file content
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"

def fetch_file_content():
    """Fetches and prints the content of Day1.py."""
    headers = {"Authorization": f"token {TOKEN}"}
    response = requests.get(API_URL, headers=headers)

    if response.status_code != 200:
        print(f"❌ Failed to fetch file: {response.json().get('message', 'Unknown error')}")
        return

    file_data = response.json()
    if "download_url" in file_data:
        download_url = file_data["download_url"]
        response = requests.get(download_url)

        if response.status_code == 200:
            print(f"\n📄 File: {FILE_PATH}\n{'=' * 50}")
            print(response.text)  # Print the content of the file
        else:
            print("❌ Failed to download the file content.")
    else:
        print("⚠️ No download URL found.")

if __name__ == "__main__":
    fetch_file_content()
