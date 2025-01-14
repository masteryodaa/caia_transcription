import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Azure OpenAI API key and endpoint
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_id = "whisper"  # Replace with your actual deployment name
version = "2024-06-01"
audio_file_path = "./a.wav"  # Path to your audio file

# Azure API URL
url = f"{api_endpoint}/openai/deployments/{deployment_id}/audio/transcriptions?api-version={version}"

# Headers for the API call
headers = {
    "api-key": api_key,
}

# Open the audio file for sending in the request
with open(audio_file_path, "rb") as audio_file:
    try:
        # Send the request to the Azure API with correct file attachment
        files = {
            "file": (audio_file_path, audio_file, "audio/wav")
        }
        response = requests.post(url, headers=headers, files=files)

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("Transcription Result:")
            print(result.get("text", "No text returned"))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Error during transcription: {e}")
