# uvicorn ec_server:app --reload --port 8000

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks
import os
import json
from datetime import datetime
from pydantic import BaseModel
from pydub import AudioSegment
import numpy as np
import wave
import csv
import asyncio
import websockets
from datetime import datetime
from openai import OpenAI
import re

from utils import save_transcript

with open("api_key", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

app = FastAPI()

CLAUSE_CSV_PATH = './data/clauses.csv'
FEEDBACK_JSON_PATH = './data/feedback.json'

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Websocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.asr_ws = None  # websocket to ASR backend
        self.asr_uri = "ws://localhost:9090/ws"
        self.idx = None
        self.audio_buffer = bytearray() 

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected: {len(self.active_connections)} total")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected: {len(self.active_connections)} remaining")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Failed to send message: {e}")

    async def start_asr_connection(self):
        if self.asr_ws:
            return
        self.asr_ws = await websockets.connect(self.asr_uri)
        print("ASR backend connected!")
        self.idx = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        # Optionally send config init here
        await self.asr_ws.send(
            json.dumps(
                {
                    "uid": asr_state["active_uid"],
                    "language": "en",
                    "task": "transcribe",
                    "model": "base",
                    "use_vad": True,
                    "max_clients": 4,
                    "max_connection_time": 600,
                    "send_last_n_segments": 10,
                    "no_speech_thresh": 0.45,
                    "clip_audio": False,
                    "same_output_threshold": 10,
                }
            )
        )

        async def asr_reader():
            output_path = f"./data/transcripts/{self.idx}.txt"
            os.makedirs("./data/transcripts", exist_ok=True)
            end_time = 0

            async for msg in self.asr_ws:
                try:
                    data = json.loads(msg)
                    if "segments" in data:
                        await self.broadcast({"type": "segment", "segments": data["segments"]})
                        end_time = save_transcript(output_path, data['segments'], end_time)
                except:
                    continue

        asyncio.create_task(asr_reader())

    async def handle_audio_bytes(self, data):
            self.audio_buffer += data

manager = ConnectionManager()

asr_state = {"active_uid": None}

@app.get("/")
async def index():
    return {"message": "EvolveCaption API server running..."}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    await websocket.send_json({"type": "asr_state", "uid": asr_state["active_uid"]})

    try:
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.receive" and msg.get("bytes"):
                if manager.asr_ws:
                    await manager.asr_ws.send(msg["bytes"])
                await manager.handle_audio_bytes(msg["bytes"])
            
            elif msg.get("type") == "websocket.receive" and msg.get("text"):
                data = json.loads(msg.get("text"))
                if data.get("type") == "start_asr":
                    uid = data.get("uid")
                    if asr_state["active_uid"] is None:
                        asr_state["active_uid"] = uid
                        print(f"ASR started at uid {uid}")
                        await manager.start_asr_connection()
                        await manager.broadcast({"type": "asr_start", "uid": uid})
                    else:
                        await websocket.send_json({"type": "error", "message": "ASR already started"})
                elif data.get("type") == "stop_asr":
                    uid = data.get("uid")
                    if asr_state["active_uid"] == uid:
                        asr_state["active_uid"] = None
                        print(f"ASR stopped at uid {uid}")
                        await manager.broadcast({"type": "asr_stop"})
                        if manager.asr_ws:
                            await manager.asr_ws.close()
                            manager.asr_ws = None

                        audio_path = f"./data/recordings/{manager.idx}.wav"
                        os.makedirs("./data/recordings", exist_ok=True)

                        try:
                            float_data = np.frombuffer(manager.audio_buffer, dtype=np.float32)
                            int16_data = np.int16(np.clip(float_data, -1.0, 1.0) * 32767)

                            with wave.open(audio_path, 'w') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                wf.writeframes(int16_data.tobytes())

                            print(f"Recording saved at {audio_path}")
                        except Exception as e:
                            print(f"Failed to save recording: {e}")

                        # Reset buffer
                        manager.audio_buffer = bytearray()

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# HTTP
@app.get("/clauses")
async def get_clauses():
    with open(CLAUSE_CSV_PATH, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return JSONResponse([row for row in reader])

class Correction(BaseModel):
    original: str
    corrected: str
    start: int
    end: int

@app.post("/feedback")
async def save_feedback(correction: Correction, background: BackgroundTasks):
    feedback_entry = {
        "original": correction.original,
        "feedback": correction.corrected,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(FEEDBACK_JSON_PATH):
        with open(FEEDBACK_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append and save
    data.append(feedback_entry)

    with open(FEEDBACK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


    print(f"Feedback: {correction.original} => {correction.corrected}")
    print(f"{correction}")

    await manager.broadcast({
        "type": "correction",
        "original": correction.original,
        "corrected": correction.corrected,
        "start": correction.start,
        "end": correction.end,
    })
    # background.add_task(gen_openai_clause, correction.original, correction.corrected)

    return {"status": "ok", "message": "Feedback saved"}

@app.post("/upload")
async def upload_audio(clause_id: str = Form(...), file: UploadFile = File(...)):
    os.makedirs("./data/audio", exist_ok=True)
    tmppath = os.path.join("data/audio", f"{clause_id}.webm")
    filepath = os.path.join("data/audio", f"{clause_id}.wav")

    with open(tmppath, "wb") as f:
        f.write(await file.read())

    audio = AudioSegment.from_file(tmppath, format="webm")
    audio.export(filepath, format="wav")

    print(f"Successfully saved to {filepath}!")

    os.remove(tmppath)

    updated_rows = []
    with open(CLAUSE_CSV_PATH, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] == clause_id:
                row["hasRecording"] = "true"
            updated_rows.append(row)

    with open(CLAUSE_CSV_PATH, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "hasRecording"])
        writer.writeheader()
        writer.writerows(updated_rows)



    return {"status": "ok", "filename": clause_id}

async def gen_openai_clause(original: str, corrected: str):
    print(f"Generating clause from: {original} => {corrected}")
    if corrected == "":
        corrected = original
    try:
        prompt = f"""
You are generating short spoken English clauses to help improve an automatic speech recognition (ASR) system. Based on a word that was misrecognized by ASR, your goal is to create a new clause (10–20 words) that:

- Sounds natural in daily conversation
- Contains the corrected word in a prominent, clear context
- Has similar phonetic structure to the original sentence

Original words: "{original}"
Corrected words: "{corrected}"

Generate one new clause that can be used to help the ASR model learn this correction. Just reply with the clause (no quotes, no explanation).
"""
        # Call OpenAI (use gpt-3.5 or gpt-4 as needed)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=30
        )

        new_clause = response.choices[0].message.content.strip()
        new_clause = re.sub(r"[^\w\s']|(?<!\w)'|'(?!\w)", '', new_clause)
        new_clause = new_clause[0].upper() + new_clause[1:]
        # new_clause = "A NEW CLAUSE"

        # Read existing CSV and get last id
        last_id = 0
        if os.path.exists(CLAUSE_CSV_PATH):
            with open(CLAUSE_CSV_PATH, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_id = int(rows[-1]["id"])

        # Append new clause
        new_row = {
            "id": str(last_id + 1),
            "text": new_clause,
            "hasRecording": "False"
        }

        file_exists = os.path.exists(CLAUSE_CSV_PATH)
        with open(CLAUSE_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "hasRecording"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_row)

        print(f"New clause generated and saved: {new_row}")

    except Exception as e:
        print(f"Error in gen_openai_clause: {e}")
