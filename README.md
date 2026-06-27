# Face Landmark Detection

แอปพลิเคชันตรวจจับจุด landmark บนใบหน้าแบบ real-time ผ่านเว็บเบราว์เซอร์ ขับเคลื่อนด้วย [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) และ [FastAPI](https://fastapi.tiangolo.com/) เบราว์เซอร์จะเปิดกล้องและส่งเฟรมไปยัง Python backend ผ่าน WebSocket จากนั้นแสดงจุด landmark 478 จุดทับบน video feed แบบ real-time ด้วย HTML5 Canvas

---

## ฟีเจอร์

- **Face mesh 478 จุด** — ครอบคลุม tesselation, contours และ iris
- **WebSocket streaming** — ความหน่วงต่ำ ประมวลผลทีละเฟรม
- **เปิดกล้องผ่านเบราว์เซอร์** — ไม่ต้องต่อกล้องกับ server โดยตรง
- **รองรับ Docker** — deploy ด้วยคำสั่งเดียว
- **แสดงภาพแบบ mirror** — เหมือนมองกระจก สะดวกสำหรับ selfie mode

---

## สถาปัตยกรรม

```
Browser (getUserMedia)
    │
    │  JPEG frame (base64 ผ่าน WebSocket)
    ▼
FastAPI Server  ──►  FaceLandmarkEngine (MediaPipe)
    │
    │  Landmark JSON (좌표 x, y, z แบบ normalized)
    ▼
Browser Canvas  ──►  วาด connections + landmark points
```

**ลำดับการทำงาน:**
1. เบราว์เซอร์เปิดกล้องที่ความละเอียด 800×600 แล้ว draw แต่ละเฟรมลง offscreen canvas
2. เฟรมถูก encode เป็น JPEG (quality 0.7) และส่งเป็น base64 JSON ผ่าน WebSocket
3. Server decode ภาพแล้วส่งให้ MediaPipe Face Landmarker ประมวลผลใน `IMAGE` mode
4. ส่ง JSON กลับพร้อมพิกัด landmark 478 จุด และ connection index ของแต่ละกลุ่ม
5. เบราว์เซอร์วาด tesselation, contours และ iris rings ทับบน `<video>` element

---

## โครงสร้างโปรเจกต์

```
workshop-5/
├── facial_landmark_detection.py  # FaceLandmarkEngine class (MediaPipe wrapper)
├── server.py                     # FastAPI app — HTTP + WebSocket endpoints
├── static/
│   └── index.html                # Frontend — เปิดกล้องและวาด landmark บน Canvas
├── face_landmarker.task          # MediaPipe model file (float16)
├── Dockerfile                    # Container definition
└── README.md
```

---

## ความต้องการของระบบ

- Python 3.12
- กล้องที่เบราว์เซอร์สามารถเข้าถึงได้
- เบราว์เซอร์รุ่นใหม่ที่รองรับ WebRTC (Chrome / Firefox / Safari)

---

## วิธีติดตั้งและรัน

### 1. รันบนเครื่อง (ด้วย venv)

```bash
# สร้างและเปิดใช้ virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# ติดตั้ง dependencies
pip install fastapi "uvicorn[standard]" opencv-python mediapipe==0.10.35 numpy

# ดาวน์โหลด MediaPipe model
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# รัน server
uvicorn server:app --reload
```

เปิดเบราว์เซอร์ไปที่ [http://localhost:8000](http://localhost:8000) แล้วอนุญาตการเข้าถึงกล้อง

### 2. รันด้วย Docker

```bash
# Build image
docker build -t face-landmark .

# รัน container
docker run -p 8000:8000 face-landmark
```

เปิดเบราว์เซอร์ไปที่ [http://localhost:8000](http://localhost:8000) — เบราว์เซอร์จัดการกล้องเอง ไม่ต้อง passthrough device เข้า container

---

## API

### `GET /`
ให้บริการหน้า frontend (`static/index.html`)

### `WebSocket /ws`

**Client → Server** (ส่งเฟรมจากกล้อง)
```json
{
  "image": "data:image/jpeg;base64,<base64-encoded-frame>"
}
```

**Server → Client** (ส่ง landmark กลับ)
```json
{
  "faces": [
    {
      "landmarks": [
        { "x": 0.512, "y": 0.341, "z": -0.021 },
        ...
      ]
    }
  ],
  "connections": {
    "tesselation": [[0, 1], [1, 2], ...],
    "contours":    [[0, 1], ...],
    "left_iris":   [[468, 469], ...],
    "right_iris":  [[473, 474], ...]
  }
}
```

พิกัดถูก normalize อยู่ในช่วง `[0.0, 1.0]` สัมพัทธ์กับขนาดเฟรม หากต้องการพิกัดพิกเซลให้คูณ `x` ด้วยความกว้าง canvas และ `y` ด้วยความสูง canvas

---

## การตั้งค่า

| ค่าคงที่ | ไฟล์ | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|---|
| `MODEL_PATH` | `facial_landmark_detection.py` | `face_landmarker.task` | path ของ MediaPipe model |
| `num_faces` | `facial_landmark_detection.py` | `1` | จำนวนใบหน้าสูงสุดที่ detect |
| `min_face_detection_confidence` | `facial_landmark_detection.py` | `0.5` | ค่าความเชื่อมั่นขั้นต่ำในการ detect |
| `min_tracking_confidence` | `facial_landmark_detection.py` | `0.5` | ค่าความเชื่อมั่นขั้นต่ำในการ track |
| ความละเอียดวิดีโอ | `static/index.html` | `800×600` | ขนาดภาพจากกล้อง |
| JPEG quality | `static/index.html` | `0.7` | คุณภาพการบีบอัดเฟรม (0.0–1.0) |

---

## Dependencies

| แพ็กเกจ | เวอร์ชัน | หน้าที่ |
|---|---|---|
| `mediapipe` | 0.10.35 | โมเดลตรวจจับ facial landmark |
| `fastapi` | 0.138.1 | Web framework + WebSocket server |
| `uvicorn` | 0.49.0 | ASGI server |
| `opencv-python` | 4.13.0.92 | decode ภาพและแปลง color space |
| `numpy` | 2.5.0 | จัดการ frame buffer |

---

## License

MIT — ดูรายละเอียดที่ [LICENSE](LICENSE)
