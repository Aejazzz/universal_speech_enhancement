from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import load_config
from backend.app.pipeline import EnhancementPipeline
from backend.app.schemas import EnhancementResponse


app = FastAPI(title="Universal Speech Enhancement Policy Learning")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = load_config()
pipeline = EnhancementPipeline(config)
Path(config.system.output_root).mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=config.system.output_root), name="outputs")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/enhance", response_model=EnhancementResponse)
async def enhance(file: UploadFile = File(...)) -> EnhancementResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac"}:
        raise HTTPException(status_code=400, detail="Only wav/mp3/flac supported.")
    incoming = Path("outputs/uploads")
    incoming.mkdir(parents=True, exist_ok=True)
    input_path = incoming / (file.filename or "input.wav")
    with input_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    result = pipeline.run(str(input_path))
    return EnhancementResponse(**result)
