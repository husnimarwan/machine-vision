r"""
Web app sederhana untuk deteksi objek pakai YOLO.

Sumber input yang didukung:
  - gambar (upload)
  - video (upload, deteksi realtime saat diputar)
  - kamera browser (getUserMedia, butuh izin browser)
  - kamera server (OpenCV, tanpa izin browser)

Jalankan:
    .\.venv\Scripts\python.exe app.py
lalu buka http://127.0.0.1:5000
"""

from __future__ import annotations

import atexit
import base64
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from ultralytics import YOLO

try:  # opsional: hanya dipakai untuk link YouTube
    import yt_dlp
except ImportError:  # pragma: no cover
    yt_dlp = None

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MODEL_PATH = BASE_DIR / "yolo26n.pt"
MODEL_DIR = BASE_DIR / "models"          # model .pt hasil upload user
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CONF_THRESHOLD = 0.25

# --- Batas upload gambar (video sengaja TANPA batas ukuran) ---
MAX_IMAGE_BYTES = 16 * 1024 * 1024   # 16 MB
JPEG_QUALITY = 80                    # kualitas frame hasil anotasi

# Batas frame base64 untuk deteksi per-frame (video & kamera). Frame kamera
# yang dikirim browser hanya ~100 KB, jadi batas ini cuma jaring pengaman.
MAX_FRAME_B64_BYTES = 8 * 1024 * 1024  # 8 MB

# --- Model YOLO: default + model .pt yang diupload user ---
MAX_MODEL_BYTES = 512 * 1024 * 1024  # 512 MB
MAX_LOADED_MODELS = 6                # batasi RAM: model terdaftar maksimal

app = Flask(__name__)
# None = tidak ada batas body request, jadi video berapa pun ukurannya diterima.
app.config["MAX_CONTENT_LENGTH"] = None

# Model default di-load sekali saat server start (bukan tiap request)
model = YOLO(str(MODEL_PATH))

MODELS: dict[str, "YOLO"] = {"default": model}
MODEL_META: dict[str, dict] = {
    "default": {
        "id": "default",
        "name": MODEL_PATH.name,
        "size": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0,
    }
}
MODELS_LOCK = threading.Lock()


def resolve_model(model_id: str | None):
    """Ambil model YOLO sesuai id dari frontend ('default' = yolo26n.pt)."""
    if not model_id or model_id == "default":
        return model
    with MODELS_LOCK:
        loaded = MODELS.get(model_id)
    if loaded is None:
        raise ValueError(
            "Model tidak tersedia lagi di server. Pilih ulang model di tab ini."
        )
    return loaded


def allowed_file(filename: str, extensions: set[str]) -> bool:
    return Path(filename).suffix.lower() in extensions


def parse_conf(value: str | None) -> float:
    try:
        conf = float(value) if value is not None else CONF_THRESHOLD
    except (TypeError, ValueError):
        conf = CONF_THRESHOLD
    return min(max(conf, 0.05), 0.95)


# ---------------------------------------------------------------------------
# Deteksi video REALTIME (streaming).
#
# Tidak ada proses background dan tidak ada batas durasi/frame. Frame video
# dikirim satu per satu oleh browser, dideteksi saat itu juga, lalu hasilnya
# dikirim balik sebagai JPEG. Jadi deteksi berjalan persis selama video
# diputar -- persis seperti `yolo predict source=video.mp4`.
#
# State per-sesi (file video mana + confidence) disimpan di memori.
# ---------------------------------------------------------------------------
VIDEO_SESSIONS: dict[str, dict] = {}
VIDEO_SESSIONS_LOCK = threading.Lock()
MAX_SESSIONS = 64  # cegah memory leak kalau sesi tidak pernah ditutup

# session_id = uuid4().hex (32 karakter hex). Dipakai juga untuk memvalidasi
# parameter URL sebelum dipakai sebagai pola glob di disk.
SESSION_ID_RE = re.compile(r"[0-9a-f]{32}")


def put_session(session_id: str, data: dict) -> None:
    with VIDEO_SESSIONS_LOCK:
        if len(VIDEO_SESSIONS) >= MAX_SESSIONS:
            # Buang sesi paling lama
            oldest = next(iter(VIDEO_SESSIONS))
            VIDEO_SESSIONS.pop(oldest, None)
        VIDEO_SESSIONS[session_id] = data


def get_session(session_id: str) -> dict | None:
    with VIDEO_SESSIONS_LOCK:
        return VIDEO_SESSIONS.get(session_id)


def drop_session(session_id: str) -> dict | None:
    with VIDEO_SESSIONS_LOCK:
        return VIDEO_SESSIONS.pop(session_id, None)


def summarize_result(result) -> tuple[list[dict], dict[str, int]]:
    """Ambil daftar deteksi + hitungan per kelas dari satu hasil YOLO."""
    boxes = result.boxes
    names = result.names
    detections: list[dict] = []
    counts: dict[str, int] = {}

    if boxes is not None and len(boxes) > 0:
        for cls_id, score, xyxy in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            label = names[int(cls_id)]
            detections.append(
                {
                    "label": label,
                    "confidence": round(float(score), 3),
                    "box": [round(float(v), 1) for v in xyxy],
                }
            )
            counts[label] = counts.get(label, 0) + 1

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, counts


# ---------------------------------------------------------------------------
# Helper video: streaming file & pembacaan frame acak (untuk seek).
# ---------------------------------------------------------------------------
def video_path_for(session_id: str) -> Path | None:
    """Path file video milik sebuah sesi.

    Sesi disimpan di memori, jadi ikut hilang kalau server restart (misalnya
    auto-reload saat kode diubah). Karena nama file selalu
    `<session_id>.<ext>`, file-nya masih bisa ditemukan di disk sehingga video
    tetap bisa diputar tanpa upload ulang.
    """
    if not session_id or not SESSION_ID_RE.fullmatch(session_id):
        return None

    session = get_session(session_id)
    if session is not None:
        path = UPLOAD_DIR / session["filename"]
        if path.exists():
            return path

    for candidate in UPLOAD_DIR.glob(f"{session_id}.*"):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            return candidate
    return None


def read_frame_at(path: Path, index: int):
    """Baca frame ke-`index` (0-based). None kalau gagal/di luar rentang."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def probe_video(path: Path) -> dict:
    """Ambil metadata video tanpa memproses isinya."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Video tidak bisa dibuka. Coba format lain (MP4).")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 240:
            fps = 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if frames <= 0:
        raise ValueError("Jumlah frame video tidak terbaca. Coba format MP4 (H.264).")

    return {
        "frames": frames,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "duration": round(frames / fps, 2),
    }


@app.errorhandler(413)
def too_large(_exc):
    # Aplikasi tidak membatasi ukuran upload; ini hanya jaga-jaga kalau ada
    # reverse proxy / server lain di depan Flask yang membatasi body request.
    return jsonify({"error": "File terlalu besar untuk diproses server."}), 413


@app.errorhandler(404)
def not_found(err):
    # Endpoint /api/* harus selalu balas JSON, jangan halaman HTML,
    # supaya frontend tidak gagal saat JSON.parse.
    if request.path.startswith("/api/"):
        return jsonify({"error": f"Endpoint tidak ditemukan: {request.path}"}), 404
    return err, 404


@app.errorhandler(500)
def server_error(err):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Terjadi kesalahan pada server."}), 500
    return err, 500


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/models/upload")
def model_upload():
    """Terima file model .pt dari PC user, muat, lalu kembalikan id-nya.

    Model disimpan di folder `models/` dan tetap ter-load di memori selama
    server hidup, supaya request deteksi berikutnya tinggal pakai id.
    """
    file = request.files.get("model")
    if file is None or not file.filename:
        return jsonify({"error": "File model tidak ditemukan."}), 400
    if not file.filename.lower().endswith(".pt"):
        return jsonify({"error": "File model harus berekstensi .pt"}), 400

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename).name)[-80:]
    path = MODEL_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    file.save(path)

    size = path.stat().st_size
    if size == 0:
        path.unlink(missing_ok=True)
        return jsonify({"error": "File model kosong."}), 400
    if size > MAX_MODEL_BYTES:
        path.unlink(missing_ok=True)
        return jsonify({"error": "Ukuran model melebihi 512 MB."}), 413

    try:
        loaded = YOLO(str(path))
    except Exception:
        path.unlink(missing_ok=True)
        return jsonify({"error": "File itu bukan model YOLO (.pt) yang valid."}), 400

    model_id = uuid.uuid4().hex
    meta = {"id": model_id, "name": file.filename, "size": size}
    with MODELS_LOCK:
        MODELS[model_id] = loaded
        MODEL_META[model_id] = meta
        # Batasi jumlah model yang tetap di RAM (model terlama dilepas)
        extra = [mid for mid in MODELS if mid != "default"]
        while len(extra) > MAX_LOADED_MODELS:
            oldest = extra.pop(0)
            MODELS.pop(oldest, None)
            MODEL_META.pop(oldest, None)

    return jsonify(meta), 201


@app.post("/api/detect")
def detect():
    if "image" not in request.files:
        return jsonify({"error": "File gambar tidak ditemukan."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Nama file kosong."}), 400
    if not allowed_file(file.filename, IMAGE_EXTENSIONS):
        return jsonify({"error": "Format harus JPG, PNG, BMP, atau WEBP."}), 400

    try:
        chosen_model = resolve_model(request.form.get("model"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Nama unik supaya file tidak saling menimpa
    stem = uuid.uuid4().hex
    ext = Path(file.filename).suffix.lower()
    input_path = UPLOAD_DIR / f"{stem}{ext}"
    file.save(input_path)

    if input_path.stat().st_size > MAX_IMAGE_BYTES:
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "Ukuran gambar melebihi 16 MB."}), 400

    conf = parse_conf(request.form.get("conf"))

    start = time.perf_counter()
    results = chosen_model.predict(source=str(input_path), conf=conf, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = results[0]

    # Simpan gambar beranotasi
    result_name = f"{stem}.jpg"
    result.save(filename=str(RESULT_DIR / result_name))

    detections, counts = summarize_result(result)

    return jsonify(
        {
            "original": f"/static/uploads/{input_path.name}",
            "annotated": f"/static/results/{result_name}",
            "total": len(detections),
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "detections": detections,
            "inference_ms": round(elapsed_ms, 1),
            "conf": conf,
        }
    )


@app.post("/api/video/upload")
def video_upload():
    """Terima file video, simpan, kembalikan metadata + URL streaming."""
    if "video" not in request.files:
        return jsonify({"error": "File video tidak ditemukan."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Nama file kosong."}), 400
    if not allowed_file(file.filename, VIDEO_EXTENSIONS):
        return jsonify(
            {"error": "Format harus MP4, AVI, MOV, MKV, WEBM, atau M4V."}
        ), 400

    stem = uuid.uuid4().hex
    ext = Path(file.filename).suffix.lower()
    input_path = UPLOAD_DIR / f"{stem}{ext}"
    file.save(input_path)

    size_bytes = input_path.stat().st_size
    if size_bytes == 0:
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "File video kosong."}), 400

    try:
        info = probe_video(input_path)
    except ValueError as exc:
        input_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400

    session_id = stem
    put_session(
        session_id,
        {"filename": input_path.name, "frames": info["frames"], "fps": info["fps"]},
    )

    return jsonify(
        {
            "session_id": session_id,
            "src": f"/api/video/file/{session_id}",
            **info,
            "size": size_bytes,
        }
    ), 201


@app.get("/api/video/file/<session_id>")
def video_file(session_id: str):
    """Serve file video untuk elemen <video> (mendukung Range request)."""
    path = video_path_for(session_id)
    if path is None:
        return jsonify({"error": "Video tidak ditemukan."}), 404
    # conditional=True -> Flask mengaktifkan dukungan HTTP Range (seek)
    return send_file(path, conditional=True)


def decode_data_url(data_url: str):
    """Ubah 'data:image/jpeg;base64,...' jadi array BGR OpenCV."""
    if not data_url:
        return None
    payload = data_url.split(",", 1)[-1]
    try:
        raw = base64.b64decode(payload, validate=False)
        buffer = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (ValueError, TypeError):
        return None


def detect_frame(payload: dict):
    """Deteksi SATU frame. Dipakai oleh input VIDEO dan input KAMERA.

    Browser mengirim frame yang SEDANG diputar / ditangkap kamera (base64 JPEG
    dari <canvas>), server menjalankan YOLO saat itu juga, lalu mengembalikan
    daftar deteksi (opsional frame beranotasi). Ini yang membuat deteksi terasa
    realtime: tidak ada pemrosesan di belakang layar, jadi tidak perlu batas
    durasi/frame.

    Body JSON:
      - image        : data URL frame (wajib, kalau tidak ada -> fallback baca file)
      - frame        : index frame (info saja)
      - conf         : confidence threshold
      - model        : id model YOLO (opsional, default yolo26n.pt)
      - session_id   : id sesi video (opsional, untuk fallback & metadata)
      - return_image : kirim balik frame beranotasi? (default true)
    """
    conf = parse_conf(payload.get("conf"))

    try:
        chosen_model = resolve_model(payload.get("model"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    image_payload = payload.get("image") or ""
    if isinstance(image_payload, str) and len(image_payload) > MAX_FRAME_B64_BYTES:
        return jsonify({"error": "Frame terlalu besar untuk diproses."}), 413

    try:
        index = int(payload.get("frame", 0))
    except (TypeError, ValueError):
        index = 0
    index = max(0, index)

    # 1) Frame dari browser (jalur utama)
    frame = decode_data_url(image_payload)

    # 2) Fallback: baca frame dari file berdasarkan index
    if frame is None:
        session_id = str(payload.get("session_id", "")).strip()
        path = video_path_for(session_id) if session_id else None
        if path is None:
            return jsonify(
                {"error": "Frame tidak terkirim dan sesi video tidak ditemukan."}
            ), 400
        frame = read_frame_at(path, index)
        if frame is None:
            return jsonify({"error": f"Frame {index} tidak bisa dibaca."}), 400

    # Ukuran frame yang benar-benar dideteksi. Browser memakai ini untuk
    # menskalakan koordinat box ke ukuran canvas tampilan.
    frame_height, frame_width = frame.shape[:2]

    start = time.perf_counter()
    result = chosen_model.predict(frame, conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections, counts = summarize_result(result)

    # Browser menggambar bounding box sendiri di atas frame video yang sedang
    # diputar (lihat templates/index.html), jadi gambar beranotasi hanya dibuat
    # kalau memang diminta. Tanpa plot + encode JPEG, tiap request jauh lebih
    # ringan sehingga deteksi saat video play bisa jalan lebih cepat.
    want_image = bool(payload.get("return_image", True))
    annotated_url = None
    if want_image:
        annotated = result.plot()
        ok, buffer = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            return jsonify({"error": "Gagal meng-encode frame hasil deteksi."}), 500
        annotated_url = (
            "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")
        )

    return jsonify(
        {
            "frame": index,
            "image": annotated_url,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "total_objects": len(detections),
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "detections": detections,
            "inference_ms": round(elapsed_ms, 1),
            "conf": conf,
        }
    )


@app.post("/api/video/detect")
def video_detect():
    """Deteksi frame video (lihat detect_frame)."""
    return detect_frame(request.get_json(silent=True) or {})


@app.post("/api/camera/detect")
def camera_detect():
    """Deteksi frame kamera (lihat detect_frame).

    Frame bisa datang dari kamera browser (getUserMedia) atau dari kamera
    server (lihat /api/camera/stream). Deteksinya sama saja: satu frame masuk,
    daftar objek keluar.
    """
    return detect_frame(request.get_json(silent=True) or {})


# ---------------------------------------------------------------------------
# KAMERA SERVER-SIDE (OpenCV)
#
# Browser bisa menolak / tidak menampilkan permintaan izin kamera (kebijakan
# Chrome, Windows privacy, ekstensi, dsb). Mode ini membuka kamera langsung
# dari server lewat OpenCV, jadi TIDAK butuh izin browser sama sekali.
#
# Server menangkap frame di thread sendiri, meng-encode JPEG, lalu browser
# menariknya sebagai stream MJPEG:
#     <img src="/api/camera/stream/<session_id>">
# Frame yang tampil tetap dikirim balik ke /api/camera/detect untuk dideteksi.
# ---------------------------------------------------------------------------
CAMERA_STREAM_MAX_WIDTH = 960    # frame stream diperkecil (hemat bandwidth)
CAMERA_JPEG_QUALITY = 75
CAMERA_PROBE_LIMIT = 4           # cek index kamera 0..3
CAMERA_PROBE_TTL = 120.0         # cache hasil probe (detik)
MAX_CAMERA_SESSIONS = 4

CAMERA_BACKENDS = (
    [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if hasattr(cv2, "CAP_DSHOW") and hasattr(cv2, "CAP_MSMF")
    else [cv2.CAP_ANY]
)

CAMERA_SESSIONS: dict[str, "CameraSession"] = {}
CAMERA_SESSIONS_LOCK = threading.Lock()
CAMERA_PROBE_CACHE: dict = {"at": 0.0, "cameras": []}
CAMERA_PROBE_LOCK = threading.Lock()

# --- Input URL (gambar / video / YouTube / IP camera) ---
URL_OPEN_TIMEOUT_MS = 10000
URL_READ_TIMEOUT_MS = 10000
MAX_URL_LENGTH = 2048
MAX_URL_IMAGE_BYTES = 16 * 1024 * 1024
URL_SCHEMES = {"http", "https", "rtsp", "rtmp", "rtmps", "udp"}

IMAGE_URL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_URL_EXTENSIONS = {
    ".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".ts", ".m3u8", ".mpd",
}

YOUTUBE_HOSTS = (
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
)

CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def open_camera_capture(index: int):
    """Buka kamera index tertentu dengan backend Windows yang tersedia."""
    for backend in CAMERA_BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def open_url_capture(url: str):
    """Buka sumber URL (video / HLS / RTSP / YouTube) lewat FFmpeg.

    Timeout dipasang supaya link yang mati tidak menggantung request terlalu
    lama (mis. RTSP yang tidak menjawab).
    """
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        URL_OPEN_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        URL_READ_TIMEOUT_MS,
    ]
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    except (cv2.error, TypeError):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    cap.release()
    return None


class CameraSession:
    """Satu sumber frame yang ditangkap server di thread sendiri.

    Sumbernya bisa kamera lokal (index) atau URL stream (YouTube / RTSP /
    HLS / video langsung). Semuanya diubah jadi frame JPEG lalu ditarik
    browser sebagai MJPEG.
    """

    def __init__(
        self,
        index: int | None = None,
        url: str | None = None,
        label: str = "",
    ):
        self.index = index
        self.url = url
        self.label = label
        self.cap = None
        self.thread: threading.Thread | None = None
        self.running = False
        self.width = 0
        self.height = 0
        self.jpeg: bytes | None = None
        self.seq = 0
        self.created_at = time.time()
        self.readers = 0          # berapa client MJPEG yang sedang membaca
        self.cond = threading.Condition()

    def _open(self):
        if self.url:
            return open_url_capture(self.url)
        return open_camera_capture(int(self.index or 0))

    def start(self) -> None:
        cap = self._open()
        if cap is None:
            if self.url:
                raise ValueError(
                    "Sumber tidak bisa dibuka. Pastikan link bisa diakses dari "
                    "server (bukan halaman HTML biasa) dan tidak butuh login."
                )
            raise ValueError(
                f"Kamera {self.index} tidak bisa dibuka. Mungkin sedang dipakai "
                "aplikasi lain (Zoom / OBS / Meet) atau tidak terpasang."
            )

        if not self.url:
            # Minta resolusi yang layak untuk deteksi (diabaikan kalau tak didukung)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            if self.url:
                raise ValueError(
                    "Sumber terbuka tapi tidak mengirim gambar. Coba link lain "
                    "(mis. .m3u8 / rtsp:// langsung, bukan halaman pemutar)."
                )
            raise ValueError(
                f"Kamera {self.index} terbuka tapi tidak mengirim gambar. "
                "Coba kamera lain."
            )

        self.cap = cap
        self.height, self.width = frame.shape[:2]
        self._encode(frame)
        self.running = True
        name = f"url-{self.index}" if self.url else f"camera-{self.index}"
        self.thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self.thread.start()

    def _encode(self, frame) -> None:
        height, width = frame.shape[:2]
        if width > CAMERA_STREAM_MAX_WIDTH:
            scale = CAMERA_STREAM_MAX_WIDTH / width
            frame = cv2.resize(
                frame,
                (CAMERA_STREAM_MAX_WIDTH, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_JPEG_QUALITY]
        )
        if not ok:
            return
        with self.cond:
            self.jpeg = buffer.tobytes()
            self.seq += 1
            self.cond.notify_all()

    def _loop(self) -> None:
        # Ikuti fps sumber supaya video file (bukan live) tidak diproses
        # secepat CPU lalu "membeku" di frame terakhir.
        interval = 0.0
        try:
            fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
            if 1.0 < fps <= 120.0:
                interval = 1.0 / fps
        except (cv2.error, TypeError, ValueError):
            interval = 0.0

        last = time.perf_counter()
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            self._encode(frame)

            if interval:
                delay = interval - (time.perf_counter() - last)
                if delay > 0:
                    time.sleep(delay)
            last = time.perf_counter()

    def wait_frame(self, last_seq: int, timeout: float = 2.0):
        """Tunggu frame baru; kembalikan (jpeg, seq) terakhir kalau timeout."""
        with self.cond:
            if self.seq == last_seq:
                self.cond.wait(timeout)
            return self.jpeg, self.seq

    def stop(self) -> None:
        self.running = False
        with self.cond:
            self.cond.notify_all()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def put_camera_session(session_id: str, session: CameraSession) -> None:
    with CAMERA_SESSIONS_LOCK:
        if len(CAMERA_SESSIONS) >= MAX_CAMERA_SESSIONS:
            oldest_id = next(iter(CAMERA_SESSIONS))
            oldest = CAMERA_SESSIONS.pop(oldest_id, None)
            if oldest is not None:
                oldest.stop()
        CAMERA_SESSIONS[session_id] = session


def get_camera_session(session_id: str) -> CameraSession | None:
    if not session_id or not SESSION_ID_RE.fullmatch(session_id):
        return None
    with CAMERA_SESSIONS_LOCK:
        return CAMERA_SESSIONS.get(session_id)


def drop_camera_session(session_id: str) -> CameraSession | None:
    with CAMERA_SESSIONS_LOCK:
        return CAMERA_SESSIONS.pop(session_id, None)


def probe_server_cameras(force: bool = False) -> list[dict]:
    """Cari kamera yang bisa dibuka server (index + resolusi).

    Hasil di-cache sebentar karena membuka kamera untuk probe butuh ~0.5-1 detik
    per device dan sempat menyalakan LED kamera.
    """
    now = time.time()
    with CAMERA_PROBE_LOCK:
        cached = CAMERA_PROBE_CACHE["cameras"]
        if not force and cached and now - CAMERA_PROBE_CACHE["at"] < CAMERA_PROBE_TTL:
            return list(cached)

    with CAMERA_SESSIONS_LOCK:
        busy = {s.index for s in CAMERA_SESSIONS.values() if s.index is not None}

    found: list[dict] = []
    for index in range(CAMERA_PROBE_LIMIT):
        if index in busy:
            # Jangan probe kamera yang sedang dipakai sesi aktif
            found.append({"index": index, "width": 0, "height": 0, "busy": True})
            continue
        cap = open_camera_capture(index)
        if cap is None:
            continue
        try:
            # Samakan dengan resolusi yang dipakai saat sesi dibuka, supaya
            # label di dropdown tidak menyesatkan.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            found.append(
                {"index": index, "width": int(width), "height": int(height)}
            )
        finally:
            cap.release()

    with CAMERA_PROBE_LOCK:
        CAMERA_PROBE_CACHE["at"] = time.time()
        CAMERA_PROBE_CACHE["cameras"] = found
    return found


def camera_slots(force_probe: bool = False) -> list[dict]:
    """Daftar slot kamera server (index 0..CAMERA_PROBE_LIMIT-1).

    Tanpa probe hasilnya instan: resolusi hanya terisi kalau kamera itu pernah
    dibuka/dipindai. `force_probe=True` memindai semua index (lambat, ~1 detik
    per kamera) dan dipakai tombol muat ulang.
    """
    if force_probe:
        probe_server_cameras(force=True)

    with CAMERA_PROBE_LOCK:
        known = {cam["index"]: cam for cam in CAMERA_PROBE_CACHE["cameras"]}
    with CAMERA_SESSIONS_LOCK:
        busy = {s.index for s in CAMERA_SESSIONS.values() if s.index is not None}

    slots: list[dict] = []
    for index in range(CAMERA_PROBE_LIMIT):
        info = known.get(index, {"index": index, "width": 0, "height": 0})
        slot = {
            "index": index,
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
        }
        if index in busy:
            slot["busy"] = True
        slots.append(slot)
    return slots


def remember_camera_resolution(index: int, width: int, height: int) -> None:
    """Simpan resolusi kamera yang baru dibuka, supaya label dropdown akurat."""
    with CAMERA_PROBE_LOCK:
        known = [
            cam for cam in CAMERA_PROBE_CACHE["cameras"] if cam.get("index") != index
        ]
        known.append({"index": index, "width": int(width), "height": int(height)})
        CAMERA_PROBE_CACHE["cameras"] = sorted(known, key=lambda cam: cam["index"])
        CAMERA_PROBE_CACHE["at"] = time.time()


@app.get("/api/camera/list")
def camera_list():
    """Daftar kamera yang bisa dibuka server (tanpa izin browser).

    Cepat: tidak membuka device. Pakai `?refresh=1` untuk memindai ulang
    (lebih lambat, tapi mengisi resolusi semua kamera).
    """
    force = request.args.get("refresh") in ("1", "true", "yes")
    return jsonify({"cameras": camera_slots(force_probe=force)})


@app.post("/api/camera/open")
def camera_open():
    """Buka kamera server dan mulai menangkap frame."""
    payload = request.get_json(silent=True) or {}
    try:
        index = int(payload.get("index", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Index kamera tidak valid."}), 400
    if index < 0 or index >= CAMERA_PROBE_LIMIT:
        return jsonify({"error": "Index kamera di luar jangkauan."}), 400

    session_id = uuid.uuid4().hex
    session = CameraSession(index)
    try:
        session.start()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    put_camera_session(session_id, session)
    remember_camera_resolution(index, session.width, session.height)
    return jsonify(
        {
            "session_id": session_id,
            "index": index,
            "width": session.width,
            "height": session.height,
            "stream": f"/api/camera/stream/{session_id}",
        }
    ), 201


@app.get("/api/camera/stream/<session_id>")
def camera_stream(session_id: str):
    """Stream MJPEG frame kamera server (untuk elemen <img>)."""
    session = get_camera_session(session_id)
    if session is None:
        return jsonify({"error": "Sesi kamera tidak ditemukan."}), 404

    def generate():
        with session.cond:
            session.readers += 1
        try:
            last_seq = -1
            while True:
                jpeg, seq = session.wait_frame(last_seq)
                if jpeg is None:
                    break
                if seq == last_seq:
                    if not session.running:
                        break
                    continue
                last_seq = seq
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
        finally:
            with session.cond:
                session.readers = max(0, session.readers - 1)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


def _close_camera_session(session_id: str) -> bool:
    session = drop_camera_session(session_id)
    if session is None:
        return False
    session.stop()
    # Hasil probe lama tidak valid lagi (device baru saja dilepas)
    with CAMERA_PROBE_LOCK:
        CAMERA_PROBE_CACHE["at"] = 0.0
    return True


@app.delete("/api/camera/<session_id>")
def camera_close(session_id: str):
    """Tutup kamera server (lepaskan device)."""
    if not _close_camera_session(session_id):
        return jsonify({"error": "Sesi kamera tidak ditemukan."}), 404
    return jsonify({"closed": session_id})


@app.post("/api/camera/close")
def camera_close_beacon():
    """Tutup kamera server lewat POST.

    Dipakai navigator.sendBeacon() saat halaman ditutup / di-reload, karena
    beacon tidak bisa memakai method DELETE. Tanpa ini, sesi kamera bisa
    tertinggal dan device tetap "dipakai" sampai server restart.
    """
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id", "")).strip()
    _close_camera_session(session_id)
    return jsonify({"closed": session_id or None})


def _camera_watchdog_loop() -> None:
    """Tutup sesi kamera yang tidak lagi dibaca client.

    Kalau tab browser ditutup / reload tanpa sempat mengirim perintah tutup,
    sesi akan tetap memegang kamera. Sesi seperti itu (tidak ada pembaca
    MJPEG) dilepas otomatis supaya kamera tidak "terkunci".
    """
    while True:
        time.sleep(5.0)
        now = time.time()
        stale: list[str] = []
        with CAMERA_SESSIONS_LOCK:
            for sid, session in CAMERA_SESSIONS.items():
                if session.readers == 0 and now - session.created_at > 12.0:
                    stale.append(sid)
        for sid in stale:
            _close_camera_session(sid)


threading.Thread(
    target=_camera_watchdog_loop, name="camera-watchdog", daemon=True
).start()


@atexit.register
def _release_all_cameras() -> None:
    with CAMERA_SESSIONS_LOCK:
        sessions = list(CAMERA_SESSIONS.values())
        CAMERA_SESSIONS.clear()
    for session in sessions:
        session.stop()


# ---------------------------------------------------------------------------
# INPUT URL: gambar, video / YouTube, dan IP camera
#
# - Gambar  : diunduh server, dideteksi sekali, hasil anotasi disimpan.
# - YouTube : di-resolve yt-dlp jadi stream URL, lalu ditangkap server.
# - IP cam  : rtsp:// / .m3u8 / stream HTTP, ditangkap server.
#
# Untuk video/stream, frame dikirim ke browser sebagai MJPEG (sama seperti
# kamera server), lalu browser mengirim frame itu ke /api/camera/detect.
# ---------------------------------------------------------------------------
def _short_error(exc: Exception, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(exc)).replace("ERROR: ", "").strip()
    return text[:limit]


def _is_youtube(host: str) -> bool:
    host = (host or "").lower()
    return any(host == name or host.endswith("." + name) for name in YOUTUBE_HOSTS)


def _sniff_content_type(url: str) -> str:
    """Tebak Content-Type tanpa mengunduh seluruh file."""
    for method, extra in (("HEAD", {}), ("GET", {"Range": "bytes=0-2048"})):
        req = urllib.request.Request(
            url,
            method=method,
            headers={"User-Agent": USER_AGENT, **extra},
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if ctype:
                    return ctype
        except Exception:
            continue
    return ""


def classify_url(url: str) -> str:
    """Tentukan jenis sumber: 'youtube', 'image', atau 'stream'."""
    if len(url) > MAX_URL_LENGTH:
        raise ValueError("Link terlalu panjang.")

    parsed = urllib.parse.urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in URL_SCHEMES:
        raise ValueError(
            "Link harus diawali http://, https://, rtsp://, atau rtmp://."
        )

    host = (parsed.hostname or "").lower()
    if _is_youtube(host):
        return "youtube"
    if scheme not in ("http", "https"):
        return "stream"

    ext = os.path.splitext(parsed.path)[1].lower()
    if ext in IMAGE_URL_EXTENSIONS:
        return "image"
    if ext in VIDEO_URL_EXTENSIONS:
        return "stream"

    ctype = _sniff_content_type(url)
    if ctype.startswith("image/"):
        return "image"
    if (
        ctype.startswith("video/")
        or "mpegurl" in ctype
        or "multipart/x-mixed-replace" in ctype
    ):
        return "stream"
    if ctype.startswith("text/html"):
        raise ValueError(
            "Link itu halaman web, bukan file gambar/video. Pakai link gambar "
            "langsung, link .m3u8, atau rtsp:// untuk kamera."
        )

    # Tidak jelas: coba sebagai gambar (pesan error-nya jelas kalau gagal)
    return "image"


def resolve_youtube(url: str) -> dict:
    """Ubah link YouTube jadi URL stream yang bisa dibuka OpenCV."""
    if yt_dlp is None:
        raise ValueError(
            "Fitur YouTube butuh paket yt-dlp. Jalankan: "
            ".\\\\.venv\\\\Scripts\\\\python.exe -m pip install yt-dlp"
        )

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Pilih video (tanpa audio) resolusi wajar supaya deteksi ringan.
        # Live stream otomatis jatuh ke format HLS lewat '/best'.
        "format": (
            "bestvideo[height<=720][ext=mp4][protocol^=http]"
            "/bestvideo[height<=720][protocol^=http]"
            "/best[height<=720][protocol^=http]"
            "/bestvideo[protocol^=http]"
            "/best[height<=720]"
            "/best"
        ),
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt_dlp.utils.DownloadError dkk
        raise ValueError(f"YouTube tidak bisa dibuka: {_short_error(exc)}") from exc

    if info.get("entries"):
        info = info["entries"][0]
    stream_url = info.get("url")
    if not stream_url:
        raise ValueError("Stream YouTube tidak ditemukan.")

    return {
        "url": stream_url,
        "title": info.get("title") or "YouTube",
        "is_live": bool(info.get("is_live")),
    }


def detect_url_image(url: str, conf: float, model_id: str | None = None):
    """Unduh gambar dari URL, deteksi sekali, simpan hasil anotasi."""
    chosen_model = resolve_model(model_id)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            raw = resp.read(MAX_URL_IMAGE_BYTES + 1)
    except Exception as exc:
        raise ValueError(f"Gambar tidak bisa diunduh: {_short_error(exc)}") from exc

    if len(raw) > MAX_URL_IMAGE_BYTES:
        raise ValueError("Ukuran gambar melebihi 16 MB.")
    if not raw:
        raise ValueError("Gambar dari link itu kosong.")

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "Link tidak bisa dibaca sebagai gambar. Kalau ini stream kamera, "
            "pakai link rtsp:// atau .m3u8."
        )

    stem = uuid.uuid4().hex
    ext = CONTENT_TYPE_EXT.get(ctype.split(";")[0].strip(), "")
    if not ext:
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        if ext not in IMAGE_URL_EXTENSIONS:
            ext = ".jpg"
    input_path = UPLOAD_DIR / f"{stem}{ext}"
    input_path.write_bytes(raw)

    start = time.perf_counter()
    result = chosen_model.predict(source=str(input_path), conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    result_name = f"{stem}.jpg"
    result.save(filename=str(RESULT_DIR / result_name))
    detections, counts = summarize_result(result)

    return jsonify(
        {
            "kind": "image",
            "source": url,
            "original": f"/static/uploads/{input_path.name}",
            "annotated": f"/static/results/{result_name}",
            "total": len(detections),
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "detections": detections,
            "inference_ms": round(elapsed_ms, 1),
            "conf": conf,
        }
    )


@app.post("/api/url/open")
def url_open():
    """Buka sumber dari link: gambar, YouTube, atau stream kamera."""
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()
    conf = parse_conf(payload.get("conf"))
    model_id = payload.get("model")

    if not url:
        return jsonify({"error": "Link belum diisi."}), 400

    try:
        kind = classify_url(url)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if kind == "image":
        try:
            return detect_url_image(url, conf, model_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    title = ""
    is_live = False
    stream_url = url

    try:
        if kind == "youtube":
            resolved = resolve_youtube(url)
            stream_url = resolved["url"]
            title = resolved["title"]
            is_live = resolved["is_live"]

        session_id = uuid.uuid4().hex
        session = CameraSession(url=stream_url, label=title or url)
        session.start()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    put_camera_session(session_id, session)
    return jsonify(
        {
            "kind": "stream",
            "source": url,
            "title": title or url,
            "is_live": is_live,
            "session_id": session_id,
            "stream": f"/api/url/stream/{session_id}",
            "width": session.width,
            "height": session.height,
        }
    ), 201


@app.get("/api/url/stream/<session_id>")
def url_stream(session_id: str):
    """Stream MJPEG dari sumber URL (YouTube / IP camera / video)."""
    return camera_stream(session_id)


@app.delete("/api/url/<session_id>")
def url_close(session_id: str):
    """Tutup sesi URL (lepaskan stream)."""
    return camera_close(session_id)


@app.post("/api/url/close")
def url_close_beacon():
    """Tutup sesi URL lewat sendBeacon saat halaman ditutup."""
    return camera_close_beacon()


@app.delete("/api/video/<session_id>")
def video_delete(session_id: str):
    """Hapus sesi + file video dari disk."""
    session = drop_session(session_id)
    # Sesi bisa sudah hilang karena server restart; file-nya tetap dicari.
    path = video_path_for(session_id)
    if session is None and path is None:
        return jsonify({"error": "Sesi tidak ditemukan."}), 404

    if path is not None:
        # Di Windows file bisa masih dikunci (browser sedang streaming atau
        # OpenCV belum melepas handle). Coba beberapa kali; kalau tetap gagal,
        # biarkan saja daripada membalas error 500.
        for _ in range(5):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.2)

    return jsonify({"deleted": session_id})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
