from fastapi import APIRouter, UploadFile, File, HTTPException
import openai
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/api")

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

@router.post("/whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Check if file is an audio file
    allowed_extensions = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.flac', '.ogg']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an audio file.")

    try:
        # Read the file content
        audio_content = await file.read()

        # Create a temporary file-like object for OpenAI
        from io import BytesIO
        audio_file = BytesIO(audio_content)
        audio_file.name = file.filename

        # Transcribe using OpenAI Whisper
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"  # Since primarily English as specified
        )

        return {"transcription": transcription.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
