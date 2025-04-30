import requests
import webbrowser
from flask import Flask, request, redirect
import base64

# GitHub OAuth Credentials
CLIENT_ID = "Ov23liS6Cf5AKCf3t3Em"  # Your GitHub Client ID
CLIENT_SECRET = "46c2635b2c8c2b48e19912f2759f668e378c64f7"  # Your GitHub Client Secret (KEEP IT SECURE)
AUTH_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
USER_API_URL = "https://api.github.com/user"
REPO_OWNER = "ExperimentedSperm"  # Your GitHub username
REPO_NAME = "PythonFromScratch"  # Repository name without file path
FILE_PATH = "Day1.py"  # File you want to fetch

# Flask App for OAuth Callback
app = Flask(__name__)

@app.route("/")
def home():
    """Redirect users to GitHub OAuth login."""
    auth_url = f"{AUTH_URL}?client_id={CLIENT_ID}&scope=repo"
    return redirect(auth_url)

def fetch_file_content(access_token):
    """Fetches content of the Day1.py file from the repository."""
    try:
        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Construct the correct API URL for the file
        file_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
        response = requests.get(file_url, headers=headers)
        
        if response.status_code == 404:
            print(f"❌ File {FILE_PATH} not found!")
            return None
            
        if response.status_code != 200:
            print(f"❌ Error accessing file: {response.status_code}")
            print(response.json().get('message', 'Unknown error'))
            return None
            
        file_data = response.json()
        if "content" in file_data:
            content = base64.b64decode(file_data["content"]).decode('utf-8')
            print(f"\n📄 Content of {FILE_PATH}:")
            print(content)
            return content
        else:
            print("❌ No content found in the file")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

@app.route("/callback")
def callback():
    """Handle OAuth callback and retrieve access token."""
    if "error" in request.args:
        error_msg = request.args.get("error_description", "Unknown error")
        print(f"❌ Authentication error: {error_msg}")
        return f"Authentication error: {error_msg}", 400
        
    code = request.args.get("code")
    if not code:
        return "No code provided", 400

    try:
        # Exchange code for an access token
        token_response = requests.post(
            TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code": code
            }
        )
        
        token_json = token_response.json()
        access_token = token_json.get("access_token")

        if not access_token:
            error_msg = token_json.get("error_description", "Failed to get access token")
            print(f"❌ {error_msg}")
            return error_msg, 400

        print("✅ Successfully authenticated!")
        
        # Fetch the file content
        content = fetch_file_content(access_token)
        if content:
            return "Successfully fetched Day1.py! Check your terminal for the content."
        else:
            return "Failed to fetch Day1.py. Check terminal for details.", 404

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, 500

if __name__ == "__main__":
    print("🚀 Starting GitHub OAuth. Open http://localhost:8000 in your browser.")
    print("⚠️ Make sure you have set the correct CLIENT_SECRET before running!")
    app.run(port=8000, debug=True)
