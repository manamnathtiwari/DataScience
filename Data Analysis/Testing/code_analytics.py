import requests
import base64
from openai import OpenAI
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GitHub Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = "ExperimentedSperm"
REPO_NAME = "PythonFromScratch"
FILE_PATH = "Day1.py"

# OpenAI Configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def fetch_github_file():
    """Fetch file content from GitHub repository."""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        file_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
        response = requests.get(file_url, headers=headers)
        
        if response.status_code == 200:
            file_data = response.json()
            content = base64.b64decode(file_data["content"]).decode('utf-8')
            return content
        else:
            print(f"Error fetching file: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def analyze_code_with_llm(code_content):
    """Analyze code using OpenAI's GPT model."""
    try:
        # Prepare the prompt for code analysis
        prompt = f"""Analyze the following Python code and provide a detailed analysis including:
        1. Code Structure and Organization
        2. Functionality and Purpose
        3. Potential Issues or Improvements
        4. Dependencies and Requirements
        5. Security Considerations
        6. Performance Analysis
        
        Code to analyze:
        {code_content}
        """
        
        # Get analysis from OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert Python code analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in LLM analysis: {str(e)}")
        return None

def generate_analytics_report(code_content, analysis):
    """Generate a comprehensive analytics report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "file_name": FILE_PATH,
        "repository": f"{REPO_OWNER}/{REPO_NAME}",
        "code_length": len(code_content),
        "analysis": analysis
    }
    
    # Save report to file
    report_filename = f"code_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=4)
    
    return report_filename

def main():
    print("🚀 Starting Code Analytics...")
    
    # Fetch code from GitHub
    print("📥 Fetching code from GitHub...")
    code_content = fetch_github_file()
    
    if not code_content:
        print("❌ Failed to fetch code from GitHub")
        return
    
    # Analyze code with LLM
    print("🤖 Analyzing code with LLM...")
    analysis = analyze_code_with_llm(code_content)
    
    if not analysis:
        print("❌ Failed to analyze code")
        return
    
    # Generate and save report
    print("📊 Generating analytics report...")
    report_file = generate_analytics_report(code_content, analysis)
    
    print(f"✅ Analysis complete! Report saved to: {report_file}")
    print("\n📝 Analysis Summary:")
    print(analysis)

if __name__ == "__main__":
    main() 