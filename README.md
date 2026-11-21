# Object Detection Web App

Web app sederhana untuk deteksi objek pada gambar, video, kamera, dan link (URL) menggunakan YOLO (Ultralytics) + Flask.

## Fitur

- **Gambar**: upload lewat klik atau drag & drop
- **Video**: deteksi realtime saat video diputar (play / pause / seek)
- **Kamera**: pilih kamera yang terpasang (webcam, kamera USB, virtual cam) lalu
  mulai deteksi realtime — ganti kamera di dropdown tanpa menghentikan deteksi
- **URL**: tempel link lalu deteksi otomatis
  - link **gambar** (JPG/PNG/WEBP/…) → hasil anotasi
  - link **video YouTube** → langsung berjalan & dideteksi realtime
  - **IP camera** (`rtsp://`, `.m3u8`, stream HTTP) → deteksi realtime
  - tanpa kontrol playback (langsung jalan)
- Preview sebelum diproses
- Slider confidence threshold (0.05 – 0.95)
- **Pilih model YOLO per tab**: default `yolo26n.pt`, atau pilih file `.pt`
  sendiri dari PC (diunggah ke server, lalu dipakai khusus tab itu)
- Tampilan side-by-side: gambar asli vs hasil anotasi (mode gambar)
- Ringkasan jumlah objek per kelas + tabel detail dengan bar visual
- Waktu inference dan laju deteksi (fps) ditampilkan

## Struktur

```
app.py                  # Backend Flask + endpoint deteksi
templates/index.html    # UI (HTML + CSS + JS, tanpa build step)
static/uploads/         # Gambar / video yang diupload (auto-dibuat)
static/results/         # Gambar hasil deteksi (auto-dibuat)
yolo26n.pt              # Model weight
requirements.txt
```

## Instalasi

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Menjalankan

```powershell
.\.venv\Scripts\python.exe app.py
```

Buka <http://127.0.0.1:5000>

## API

### Model YOLO

Semua endpoint deteksi menerima field `model` (id model). Tanpa field itu,
dipakai model default `yolo26n.pt`.

| Endpoint                  | Keterangan                                                     |
| ------------------------- | -------------------------------------------------------------- |
| `POST /api/models/upload` | multipart `model` (file `.pt`) → `{id, name, size}`; model langsung dimuat ke RAM |

Model hasil upload disimpan di folder `models/` dan tetap ter-load selama
server hidup (maksimal 6 model; yang terlama dilepas dari RAM). Setelah server
restart, model kembali ke default — pilih ulang file `.pt`-nya.

`POST /api/detect` — multipart/form-data

| Field   | Tipe | Keterangan                                   |
| ------- | ---- | -------------------------------------------- |
| `image` | file | Wajib. JPG / PNG / BMP / WEBP, maks 16 MB    |
| `conf`  | text | Opsional. Default `0.25`, di-clamp 0.05–0.95 |
| `model` | text | Opsional. Id model dari `/api/models/upload` |

Contoh:

```powershell
curl.exe -X POST -F "image=@gambar.jpg" -F "conf=0.25" http://127.0.0.1:5000/api/detect
```

Response:

```json
{
  "original": "/static/uploads/<id>.jpg",
  "annotated": "/static/results/<id>.jpg",
  "total": 17,
  "counts": { "car": 10, "person": 5, "bus": 1, "motorcycle": 1 },
  "detections": [
    { "label": "bus", "confidence": 0.909, "box": [388.5, 71.0, 638.2, 202.8] }
  ],
  "inference_ms": 32.5,
  "conf": 0.25
}
```

### Video

| Endpoint                     | Keterangan                                              |
| ---------------------------- | ------------------------------------------------------- |
| `POST /api/video/upload`     | multipart `video` → `session_id`, `src`, metadata video |
| `GET /api/video/file/<id>`   | streaming file video (mendukung Range / seek)           |
| `POST /api/video/detect`     | JSON: `image` (data URL frame), `conf`, `frame`, `session_id`, `return_image` |
| `DELETE /api/video/<id>`     | hapus sesi + file video                                 |

### Kamera

Ada **dua sumber kamera**, bisa dipilih di dropdown "Sumber kamera":

| Sumber      | Cara kerja                                        | Butuh izin browser? |
| ----------- | ------------------------------------------------- | ------------------- |
| `browser:`  | `getUserMedia()` + `enumerateDevices()` di browser | Ya                  |
| `server:`   | OpenCV membuka kamera di server, browser menarik stream MJPEG | Tidak       |

Kalau izin browser ditolak / prompt-nya tidak muncul, aplikasi otomatis
beralih ke kamera server (selama ada kamera yang bisa dibuka server). Dropdown
juga default ke kamera server supaya deteksi langsung jalan tanpa prompt izin;
kamera browser tetap bisa dipilih manual.

`POST /api/camera/detect` — JSON, body & response sama dengan `/api/video/detect`:

| Field          | Tipe   | Keterangan                                          |
| -------------- | ------ | --------------------------------------------------- |
| `image`        | string | Wajib. Data URL frame (`data:image/jpeg;base64,...`) |
| `conf`         | number | Opsional. Default `0.25`, di-clamp 0.05–0.95        |
| `model`        | string | Opsional. Id model YOLO                             |
| `return_image` | bool   | Opsional. Default `true`; UI memakai `false`        |

Endpoint kamera server:

| Endpoint                        | Keterangan                                        |
| ------------------------------- | ------------------------------------------------- |
| `GET /api/camera/list`          | Daftar slot kamera server (instan; `?refresh=1` memindai resolusi, lebih lambat) |
| `POST /api/camera/open`         | JSON `{"index": 0}` → `session_id`, `width`, `height`, `stream` |
| `GET /api/camera/stream/<id>`   | Stream MJPEG frame kamera (untuk `<img>`)         |
| `DELETE /api/camera/<id>`       | Tutup sesi kamera + lepaskan device               |

### URL

Satu endpoint untuk semua jenis link. Jenisnya ditentukan otomatis dari domain,
ekstensi, dan `Content-Type`:

| Jenis          | Contoh link                                    | Cara kerja                          |
| -------------- | ---------------------------------------------- | ----------------------------------- |
| Gambar         | `https://.../foto.jpg`                         | diunduh, dideteksi sekali           |
| YouTube        | `https://youtu.be/<id>`                        | di-resolve `yt-dlp`, lalu di-stream |
| IP camera      | `rtsp://user:pass@ip:554/stream`               | OpenCV/FFmpeg menangkap frame       |
| HLS / video    | `https://.../stream.m3u8`, `https://.../a.mp4` | OpenCV/FFmpeg menangkap frame       |

| Endpoint                        | Keterangan                                        |
| ------------------------------- | ------------------------------------------------- |
| `POST /api/url/open`            | JSON `{"url": "...", "model": "<id opsional>"}` → `kind: "image"` (hasil deteksi) atau `kind: "stream"` (`session_id`, `stream`, `title`, `width`, `height`) |
| `GET /api/url/stream/<id>`      | Stream MJPEG dari link (YouTube / IP camera / video) |
| `DELETE /api/url/<id>`          | Tutup sesi URL                                    |
| `POST /api/url/close`           | Tutup sesi URL lewat `sendBeacon` saat halaman ditutup |

Link YouTube butuh paket **yt-dlp** (sudah ada di `requirements.txt`). Deteksi
stream memakai frame MJPEG yang sama seperti kamera server, jadi ringan di
browser dan tidak butuh CORS.

## Catatan

- Model di-load sekali saat server start, bukan tiap request.
- Akses kamera lewat browser butuh *secure context*: pakai `http://127.0.0.1:5000`
  atau `http://localhost:5000`. Kalau halaman dibuka lewat IP LAN
  (`http://192.168.x.x:5000`), browser akan memblokir `getUserMedia`.
- Kalau browser tidak memunculkan permintaan izin kamera (atau izinnya diblokir):
  pakai opsi **Kamera N · server** di dropdown. Mode itu membuka kamera lewat
  OpenCV di server, jadi tidak bergantung pada izin browser.
- Nama kamera browser baru terbaca setelah izin diberikan. Tombol muat ulang
  (ikon ↻) dipakai kalau kamera dipasang/dicabut.
- Kamera (browser maupun server) otomatis dilepas saat tab Kamera ditinggalkan,
  deteksi dihentikan, atau halaman ditutup.
- Stream URL (YouTube / IP camera) juga otomatis dilepas saat tab URL
  ditinggalkan atau halaman ditutup.
- YouTube diunduh lewat server, jadi PC harus bisa mengakses YouTube. Video
  diambil resolusi ≤720p supaya deteksi tetap ringan.
- IP camera RTSP/HLS harus bisa diakses dari PC tempat server berjalan. Kalau
  kamera butuh login, tulis kredensialnya di URL (`rtsp://user:pass@ip:554/...`).
- Server Flask ini development server. Untuk production pakai `waitress` / `gunicorn`.
- File di `static/uploads/` dan `static/results/` tidak dibersihkan otomatis.
