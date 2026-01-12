# 🚗 SafeDrive: AI Vehicle Tracking & Risk Assessment System

**SafeDrive** è un sistema avanzato di Computer Vision progettato per il monitoraggio del traffico in tempo reale. Combina il rilevamento oggetti (YOLO), il tracciamento persistente con memoria visiva, il riconoscimento targhe (OCR) e un sistema di valutazione del rischio di collisione.

## ✨ Funzionalità

- **Rilevamento Veicoli:** Identifica auto, bus e camion utilizzando `YOLOv8/v11`.
- **Valutazione Rischio (State Machine):** Calcola in tempo reale lo stato di ogni veicolo:
  - 🟢 **SAFE:** Lontano o non in traiettoria.
  - 🟡 **WARNING:** Distanza media o avvicinamento.
  - 🔴 **DANGER:** Vicino e in traiettoria di collisione (calcolo TTC).
- **Memoria Visiva (TOOCM):** Re-identifica veicoli persi temporaneamente (occlusioni) confrontando istogrammi colore e posizione.
- **Riconoscimento Targhe (ALPR):** Scansiona le targhe con `EasyOCR`, utilizza un sistema di "voto" per la conferma e salva i dati su **MongoDB**.
- **Metrics Evaluation:** Script separato per calcolare metriche standard MOT (MOTA, MOTP, IDF1) usando `motmetrics`.

---

## 📂 Struttura del Progetto

Per garantire il funzionamento degli import (es. `from src...`), assicurati che la tua cartella di lavoro sia organizzata esattamente così:

```text
SafeDrive/
│
├── assets/
│   ├── videoOBS/
│   │   └── video4.mp4       # Video di input
│   ├── gt.csv               # (Opzionale) Ground Truth per i test
│   └── 0001/                # (Opzionale) Cartella frame per valutazione
│
├── src/
│   ├── __init__.py          # Importante: rende la cartella un package
│   ├── behavior/
│   │   ├── __init__.py
│   │   ├── state_machine.py # Logica stati (Safe, Warning, Danger)
│   │   └── risk_observer.py # Observer pattern per notifiche
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── db_manager.py    # Connessione a MongoDB
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── gt_loader.py     # Loader CSV Ground Truth
│   │   └── mot_evaluator.py # MotMetrics Wrapper
│   │
│   ├── input_ouput/
│   │   ├── __init__.py
│   │   └── video_facade.py  # Gestione Webcam/File Video
│   │
│   └── processing/
│       ├── __init__.py
│       ├── detector.py      # Wrapper YOLO e Tracking
│       ├── plate_recognizer.py # OCR asincrono e code
│       └── tracker_memory.py   # Memoria Visiva (TOOCM)
│
├── main.py                  # Script Principale
├── eval_run.py              # Script di Valutazione (Metrics)
├── requirements.txt         # Dipendenze
└── README.md

Installazione e Requisiti
1. Prerequisiti
Python 3.8+

MongoDB: Deve essere installato e in esecuzione locale sulla porta 27017.

2. Installazione Dipendenze
Crea un file requirements.txt o installa direttamente le librerie:pip install ultralytics opencv-python numpy pymongo easyocr motmetrics pandas scipy

3. Utilizzo
Configurazione
Apri main.py e modifica la sezione iniziale se necessario:
# main.py
video_path = "assets/videoOBS/video4.mp4"  # Metti 0 per usare la Webcam
model_name = "yolov8s.pt"                  # Il modello viene scaricato al primo avvio

4.Avvio del Sistema
Esegui il file principale dalla cartella radice del progetto: python main.py

Comandi durante l'esecuzione:
Premi q per chiudere l'applicazione.
Guarda il terminale per i log di pericolo e le targhe confermate.

📊 Valutazione (Benchmark)
Se desideri calcolare le metriche di performance (MOTA, IDF1) rispetto a un Ground Truth:

Prepara un file gt.csv (formato: frame,id,x1,y1,x2,y2).

Prepara una cartella contenente i singoli frame del video (es. assets/0001/).

Esegui lo script:
python eval_run.py --frames "assets/0001" --gt "assets/gt.csv" --model "yolov8s.pt"

📝 Note Tecniche
Database: Il sistema crea automaticamente un DB idTracking_db e una collezione tracked_objects su MongoDB.

OCR Threading: Il riconoscimento targhe gira su un thread separato per non rallentare il video.

Struttura Import: Non spostare i file fuori dalla cartella src senza aggiornare i percorsi negli import.