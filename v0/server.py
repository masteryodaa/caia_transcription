import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI() 

# List of allowed origins (frontend URLs)
origins = [
    "http://127.0.0.1:5500",  # Frontend origin
    "http://localhost:5500",   # Local development
    "http://localhost:8000",   # Server origin if needed
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Azure configuration
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
print(AZURE_ENDPOINT)
DEPLOYMENT_ID = "whisper"
API_VERSION = "2024-06-01"
chatgpt_model = "gpt-4o"
AZURE_GPT_KEY = os.getenv("AZURE_OPENAI_GPT_KEY")
AZURE_OPENAI_GPT_ENDOINT = os.getenv("AZURE_OPENAI_GPT_ENDOINT")


def chunk_audio(file_content, chunk_size=20 * 1024 * 1024):  # 20MB
    """
    Split the audio file into chunks of specified size.
    """
    chunks = []
    for i in range(0, len(file_content), chunk_size):
        chunks.append(file_content[i:i + chunk_size])

    return chunks


@app.post("/transcribe/")
async def transcribe_and_format_audio(file: UploadFile = File(...)):
    """
    Endpoint to transcribe and format audio using Azure APIs with chunking support for large files.
    """
    if not file.content_type.startswith("audio"):
        raise HTTPException(
            status_code=400, detail="Uploaded file is not an audio file.")

    transcription_url = f"{AZURE_ENDPOINT}/openai/deployments/{DEPLOYMENT_ID}/audio/transcriptions?api-version={API_VERSION}"
    formatting_url = f"{AZURE_OPENAI_GPT_ENDOINT}/openai/deployments/{chatgpt_model}/chat/completions?api-version=2024-08-01-preview"
    headers = {"api-key": AZURE_API_KEY}
    gpt_headers = {"api-key": AZURE_GPT_KEY}

    try:
        # Read the file content once
        file_content = await file.read()
        file_size = len(file_content)
        print(f"File size: {file_size} bytes")

        # If the file size is less than 20MB, process it normally
        if file_size <= 20 * 1024 * 1024:
            print("Starting transcription for smaller audio file...")
            transcription_response = requests.post(
                transcription_url,
                headers=headers,
                files={"file": (file.filename, io.BytesIO(
                    file_content), file.content_type)}
            )

            if transcription_response.status_code != 200:
                print("Error during transcription:",
                      transcription_response.text)
                return JSONResponse(content={"error": transcription_response.text}, status_code=transcription_response.status_code)

            transcription_result = transcription_response.json()
            transcribed_text = transcription_result.get(
                "text", "No transcription available.")
            print("Transcription completed:", transcribed_text)

        else:
            # For larger files, split into chunks
            print("Starting chunking process for larger file...")
            chunks = chunk_audio(file_content)
            transcribed_text = ""

            for idx, chunk in enumerate(chunks):
                print(f"Processing chunk {idx + 1}/{len(chunks)}...")

                # Send the chunk to the transcription API
                # Convert chunk into BytesIO stream
                audio_chunk = io.BytesIO(chunk)
                transcription_response = requests.post(
                    transcription_url,
                    headers=headers,
                    files={"file": (file.filename, audio_chunk,
                                    file.content_type)}
                )

                if transcription_response.status_code != 200:
                    print(
                        f"Error during transcription of chunk {idx + 1}: {transcription_response.text}")
                    return JSONResponse(content={"error": transcription_response.text}, status_code=transcription_response.status_code)

                transcription_result = transcription_response.json()
                transcribed_text += transcription_result.get("text", "")
                print(f"Chunk {idx + 1} transcription completed.")

            print("Transcription completed. Combined Text:", transcribed_text)

        # Formatting API call
        formatting_prompt = f"""
        Format the following transcription into a structured doctor-patient conversation like:
        Example 1:
        [Doctor]: Good morning, how can I help you today?
        [Patient]: I’ve been feeling a sharp pain in my lower back for the past week.
        [Doctor]: On a scale of 1 to 10, how severe is the pain?
        [Patient]: I’d say it’s around 7.

        Example 2:
        [Doctor]: Do you have any other symptoms?
        [Patient]: Yes, I’ve also been experiencing headaches and fatigue.

        Now format the following transcription:
        {transcribed_text}
        """

        print("Starting formatting process...")

        formatting_response = requests.post(
            formatting_url,
            headers=gpt_headers,
            json={
                "messages": [{"role": "user", "content": formatting_prompt}],
                "max_tokens": 500,
                "temperature": 0.5,
                "top_p": 1.0
            }
        )

        if formatting_response.status_code != 200:
            print("Error during formatting:", formatting_response.text)
            return JSONResponse(content={"error": formatting_response.text}, status_code=formatting_response.status_code)

        formatting_result = formatting_response.json()
        formatted_text = formatting_result.get("choices", [
                                               {"message": {"content": "No formatting available."}}])[0]["message"]["content"]

        print("Formatting completed. Formatted text:", formatted_text)

        return JSONResponse(content={"formatted_text": formatted_text})

    except Exception as e:
        print("An error occurred:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/summarize")
async def summarize_text(request: Request):
    """
    Endpoint to summarize text using Azure GPT.
    """
    try:
        print("Starting summarization process...")

        # Get the request body
        body = await request.json()
        text = body.get("text")

        if not text:
            raise HTTPException(status_code=400, detail="Text is required.")

        # Format the prompt for summarization
        prompt = f"Summarize the following text:\n\n{text}"

        # Define the API endpoint and headers
        url = f"{AZURE_OPENAI_GPT_ENDOINT}/openai/deployments/{chatgpt_model}/chat/completions?api-version=2024-08-01-preview"
        headers = {"api-key": AZURE_GPT_KEY}

        # Send the summarization request to GPT
        response = requests.post(
            url,
            headers=headers,
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.5,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stop": ["\n"]
            }
        )

        if response.status_code != 200:
            print("Error during summarization:", response.text)
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

        result = response.json()

        # Extract the summarized text from the response
        summary = result.get("choices", [{"message": {"content": "No summary available."}}])[
            0]["message"]["content"]

        print("Summarization completed. Summary:", summary)

        return JSONResponse(content={"text": summary})

    except Exception as e:
        print("An error occurred:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
