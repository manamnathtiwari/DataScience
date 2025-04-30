import requests
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# GitHub Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = "ExperimentedSperm"
REPO_NAME = "PythonFromScratch"
FILE_PATH = "Day1.py"

# Hugging Face Configuration
MODEL_NAME = "codellama/CodeLlama-7b-hf"  # You can also use "codellama/CodeLlama-13b-hf" for better results
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_model():
    """Initialize the model and tokenizer."""
    print("🤖 Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

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

def analyze_code_with_llm(model, tokenizer, code_content):
    """Analyze code using CodeLlama model."""
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
        
        Analysis:"""
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and return the analysis
        analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the analysis part (after the prompt)
        analysis = analysis[len(prompt):].strip()
        
        return analysis
        
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
        "model_used": MODEL_NAME,
        "analysis": analysis
    }
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Save report to file
    report_filename = f"reports/code_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=4)
    
    return report_filename

def main():
    print("🚀 Starting Code Analytics with Hugging Face...")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model()
    if not model or not tokenizer:
        print("❌ Failed to initialize model")
        return
    
    # Fetch code from GitHub
    print("📥 Fetching code from GitHub...")
    code_content = fetch_github_file()
    
    if not code_content:
        print("❌ Failed to fetch code from GitHub")
        return
    
    # Analyze code with LLM
    print("🤖 Analyzing code with CodeLlama...")
    analysis = analyze_code_with_llm(model, tokenizer, code_content)
    
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