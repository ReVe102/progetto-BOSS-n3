import cv2
import traceback

from src.input_ouput.video_facade import VideoInputFacade
from src.processing.detector import ObjectDetector
# Importiamo il Manager e l'Observer invece delle singole classi logiche
from src.behavior.risk_observer import TrackManager, ConsoleAlertObserver
from src.data.db_manager import DBManager
from src.processing.plate_recognizer import PlateRecognizer

def draw_hud(frame, tracks):
    """.
    Disegna box e testi sul frame.
    """
    for obj in tracks:
        # Nota: il detector restituisce 'bbox', assicuriamoci che l'oggetto lo abbia aggiornato
        x1, y1, x2, y2 = obj.info['bbox']
        
        #  chiediamo il colore allo stato corrente dell'oggetto
        color = obj.state.color
        state_name = obj.state.name
        
        # Disegno del box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Etichetta sfondo nero per leggibilità
        label = f"ID:{obj.id} [{state_name}]"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def main():
    # CONFIGURAZIONE
    video_path = "assets/video4.mp4"  # Sostituisci con 0 per la webcam
    model_name = "yolov8s.pt"
    
    try:
        # 1. INIZIALIZZAZIONE COMPONENTI
        video_loader = VideoInputFacade(video_path)
        # Otteniamo le dimensioni del video per i calcoli di rischi
        w, h, fps = video_loader.get_video_info()
        detector = ObjectDetector(model_name=model_name)
        
        # 2. INIZIALIZZAZIONE LOGICA COMPORTAMENTALE
        manager = TrackManager()            # Il "Cervello" che gestisce le tracce
        alert_system = ConsoleAlertObserver() # La "Voce" che urla in caso di pericolo
        
        # Colleghiamo l'observer al manager
        manager.attach(alert_system)

        # 3. INIZIALIZZAZIONE DB E OCR
        print("Connessione al database in corso...")
        try:
            db_manager = DBManager()
            print("Connessione al database stabilita con successo.")
        except Exception as e:
            print(f"ERRORE CRITICO: Impossibile connettersi al database: {e}")
            # Non ritorniamo, continuiamo senza DB se necessario, o ritorniamo se è bloccante.
            # return 

        plate_recognizer = PlateRecognizer()

        print(f"Sistema avviato. Risoluzione: {w}x{h}")

        frame_count = 0
        while True:
            # A. INPUT
            frame = video_loader.get_frame()
            if frame is None: break 
            
            frame_count += 1

            # B. PROCESSING (YOLO) 
            detections = detector.detect_and_track(frame)
            
            # C. LOGIC (Observer + State Pattern)
            # Passiamo tutto al manager. Lui aggiorna gli stati e notifica se serve.
            manager.update_tracks(detections, w, h)

            # D. OCR (Riconoscimento Targhe)
            for det in detections:
                obj_id = det['id']
                bbox = det['bbox']
                bbox_w = bbox[2] - bbox[0]
                
                # ASYNC OCR: Aggiungiamo alla coda di elaborazione
                # Eseguiamo ogni 5 frame e solo se l'oggetto è abbastanza grande
                if frame_count % 5 == 0 and bbox_w > 80:
                    plate_recognizer.add_to_queue(frame, obj_id, bbox)

            # E. RENDERING
            # Chiediamo al manager la lista degli oggetti correnti per disegnarli
            current_objects = manager.get_tracks()
            draw_hud(frame, current_objects)

            # Display
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("SafeDrive", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        video_loader.release()
        
    except Exception as e:
        print(f"Errore critico: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()