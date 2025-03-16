from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
from denoise import remove_noise

app = FastAPI()

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    # audio temporario
    input_path = "input_audio.wav"
    output_path = "output_audio.wav"
    
    with open(input_path, "wb") as f:
        f.write(file.file.read())

    # Remove ruido
    remove_noise(input_path, output_path)

    return {"message": "Áudio processado!", "output_file": output_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
