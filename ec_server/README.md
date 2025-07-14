# EvolveCaption - Backend

## Setup

Python ENV
```
conda create -n "whisper_env" python=3.9
conda activate whisper_env
```
Then, install WhisperLive

Start WhisperLive server
```
python run_server.py --port 9090 --backend faster_whisper
```

SSL
```
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes
```

Start API server

```
python -m uvicorn ec_server:app --reload --host 0.0.0.0 --port 8000 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```