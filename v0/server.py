from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Azure OpenAI configuration
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_ID = "whisper"  # Replace with your deployment name
API_VERSION = "2024-06-01"

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to transcribe audio using Azure Whisper API.
    """
    if not file.content_type.startswith("audio"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an audio file.")

    url = f"{AZURE_ENDPOINT}/openai/deployments/{DEPLOYMENT_ID}/audio/transcriptions?api-version={API_VERSION}"

    headers = {
        "api-key": AZURE_API_KEY,
    }

    try:
        # Send audio to Azure API
        response = requests.post(
            url,
            headers=headers,
            files={"file": (file.filename, await file.read(), file.content_type)}
        )

        if response.status_code == 200:
            result = response.json()
            return JSONResponse(content={"text": result.get("text", "No transcription available.")})
        else:
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
