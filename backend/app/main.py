from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from pydantic import BaseModel, HttpUrl
from backend.app.recorder import Recorder
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
from pathlib import Path
import m3u8
import requests
from typing import Optional, List
import base64
import json
from datetime import datetime

app = FastAPI(title="Surf Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
ROOT_DIR = Path(__file__).resolve().parents[2] 
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND = ROOT_DIR / "frontend"
RECORDINGS_DIR = ROOT_DIR / "recordings"

# Create recordings directory if it doesn't exist
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Recordings directory: {RECORDINGS_DIR}")

@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND / "index.html"))    


@app.on_event("startup")
async def load_model():
    global model
    print(BASE_DIR)
    model_path = BASE_DIR / "models" / "yolo8n_opt.pt"
    if model_path.exists():
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully from {model_path}")
    else:
        raise Exception("Model weights not found")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


# Request Models
class M3U8Request(BaseModel):
    url: HttpUrl
    max_frames: Optional[int] = None  # Process only N frames (for testing)
    confidence_threshold: Optional[float] = 0.5
    skip_frames: Optional[int] = 1  # Process every Nth frame (1 = all frames)


class DetectionResult(BaseModel):
    frame_number: int
    timestamp: float
    detections: List[dict]
    frame_base64: Optional[str] = None

stream_running = False


@app.websocket("/ws/detect/m3u8")
async def websocket_detect_m3u8(websocket: WebSocket):
    global stream_running
    await websocket.accept()

    if stream_running:
        await websocket.send_json({"error": "stream already running"})
        await websocket.close()
        return

    stream_running = True
    recorder = None
    cap = None

    try:
        config = json.loads(await websocket.receive_text())

        url = config["url"]
        confidence = float(config.get("confidence_threshold", 0.2))
        skip_frames = int(config.get("skip_frames", 1))

        cap = cv2.VideoCapture(url)
        frame_count = 0

        if not cap.isOpened():
            await websocket.send_json({"error": "Cannot open stream"})
            return

        recorder = Recorder(
            target_class="wave",
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            conf_threshold=confidence,
            out_dir=str(RECORDINGS_DIR),  # Use absolute path
            fps=25,
            record_seconds=30   # <-- fixed-duration recording
        )

        while True:
            ret, frame = cap.read()

            if websocket.client_state != WebSocketState.CONNECTED:
                break

            if not ret:
                await asyncio.sleep(1)
                continue

            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            results = model(frame, conf=confidence)

            # 🔴 RECORDING LOGIC (your class, untouched)
            recorder.update(frame, results)

            # Optional: draw boxes for frontend only
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)

            await websocket.send_json({
                "frame": base64.b64encode(buffer).decode(),
                "frame_number": frame_count
            })

            frame_count += 1

    except WebSocketDisconnect:
        pass

    finally:
        stream_running = False

        if recorder is not None:
            recorder.stop()

        if cap is not None:
            cap.release()

        await websocket.close()


# Mount static files BEFORE the list endpoint
app.mount(
    "/recordings",
    StaticFiles(directory=str(RECORDINGS_DIR)),
    name="recordings"
)


@app.get("/api/recordings/list")
def list_recordings():
    """List all recorded videos"""
    try:
        # Check if directory exists
        if not RECORDINGS_DIR.exists():
            print(f"❌ Recordings directory does not exist: {RECORDINGS_DIR}")
            return []
        
        # Get all mp4 files
        files = sorted(
            RECORDINGS_DIR.glob("*.mp4"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        print(f"📹 Found {len(files)} recordings in {RECORDINGS_DIR}")
        
        result = [
            {
                "filename": f.name,
                "url": f"/recordings/{f.name}",
                "size_mb": round(f.stat().st_size / 1_000_000, 2),
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            }
            for f in files
        ]
        
        return result
        
    except Exception as e:
        print(f"❌ Error listing recordings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/recordings/delete/{filename}")
def delete_recording(filename: str):
    """Delete a specific recording"""
    try:
        file_path = RECORDINGS_DIR / filename
        
        # Security check: ensure the file is within RECORDINGS_DIR
        if not file_path.resolve().is_relative_to(RECORDINGS_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete the file
        file_path.unlink()
        print(f"🗑️ Deleted recording: {filename}")
        
        return {"success": True, "message": f"Deleted {filename}"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting recording: {e}")
        raise HTTPException(status_code=500, detail=str(e))