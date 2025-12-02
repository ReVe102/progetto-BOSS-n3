import cv2
import traceback

from src.input_ouput.video_facade import VideoInputFacade
from src.processing.detector import ObjectDetector
# Importiamo il Manager e l'Observer invece delle singole classi logiche
from src.behavior.risk_observer import TrackManager, ConsoleAlertObserver

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

        print(f"Sistema avviato. Risoluzione: {w}x{h}")

        while True:
            # A. INPUT
            frame = video_loader.get_frame()
            if frame is None: break 

            # B. PROCESSING (YOLO) 
            detections = detector.detect_and_track(frame)
            
            # C. LOGIC (Observer + State Pattern)
            # Passiamo tutto al manager. Lui aggiorna gli stati e notifica se serve.
            manager.update_tracks(detections, w, h)

            # D. RENDERING
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