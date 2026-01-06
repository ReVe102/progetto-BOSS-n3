import cv2
import traceback
import os

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
   # video_path = "http://192.168.1.9:8080/video"  # Sostituisci con 0 per la webcam
    video_path = "assets/videoOBS/videoTestFinale.mp4"  # Percorso del video di input
    model_name = "yolov8s.pt"  # Modello YOLO da usare
    conf_threshold = 0.50   # Soglia di confidenza per il detector

    
    try:
        # 1. INIZIALIZZAZIONE COMPONENTI
        video_loader = VideoInputFacade(video_path)
        # Otteniamo le dimensioni del video per i calcoli di rischi
        w, h, fps = video_loader.get_video_info()
        detector = ObjectDetector(model_name=model_name,conf_threshold=conf_threshold)
        
        # 2. INIZIALIZZAZIONE LOGICA COMPORTAMENTALE
        manager = TrackManager()            # Il "Cervello" che gestisce le tracce
        alert_system = ConsoleAlertObserver() # La "Voce" che urla in caso di pericolo
        manager.attach(alert_system)   # Colleghiamo l'observer al manager

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
        
        # ID Mapping for reassignments
        id_map = {}

        #evaluator = MotEvaluator(iou_threshold=0.5, id_tag="run-1")

        print(f"Sistema avviato. Risoluzione: {w}x{h}")

        frame_count = 0
        while True:
            # A. INPUT
            frame = video_loader.get_frame()
            if frame is None: break 
            frame_count += 1

            # B. PROCESSING (YOLO)
            
            # Check for ID reassignments from PlateRecognizer
            reassignments = plate_recognizer.get_pending_reassignments()
            for old_id, new_id in reassignments:
                print(f"[MAIN] Applying reassignment: {old_id} -> {new_id}")
                
                # Update existing mappings that point to old_id to point to new_id
                for k, v in id_map.items():
                    if v == old_id:
                        id_map[k] = new_id
                
                id_map[old_id] = new_id
                # Also merge history in PlateRecognizer so it knows about the new ID
                plate_recognizer.merge_history(old_id, new_id)

            # --- OCR PLATE MAP (sync for this frame) ---
            ocr_plate_map = {}  # {(x1, y1, x2, y2): plate_text}
            # Optionally, you could maintain a shared structure between PlateRecognizer and here
            # For demo: let's assume you have a synchronous recognizer for this step
            # (If not, you need to collect confirmed plates from PlateRecognizer's history)

            # For each detection, try to get a confirmed plate from PlateRecognizer
            for det in detections if 'detections' in locals() else []:
                bbox = det['bbox']
                obj_id = det['id']
                # Try to get the most common plate for this obj_id from PlateRecognizer
                plate = None
                if obj_id in plate_recognizer.plate_history:
                    counts = plate_recognizer.plate_history[obj_id]
                    if counts:
                        from collections import Counter
                        c = Counter(counts)
                        most_common, count = c.most_common(1)[0]
                        if count >= 3:
                            plate = most_common
                if plate:
                    ocr_plate_map[bbox] = plate

            # Set the map for the detector to use in this frame
            detector.ocr_plate_map = ocr_plate_map if ocr_plate_map else None

            # Now run detection/tracking
            detections = detector.detect_and_track(frame)

            # Apply ID mapping to detections
            for det in detections:
                # If this ID has been remapped, update it
                if det['id'] in id_map:
                    det['id'] = id_map[det['id']]

            # C. LOGIC (Observer + State Pattern)
            manager.update_tracks(detections, w, h)
            

            # D. OCR (Riconoscimento Targhe)
            for det in detections:
                obj_id = det['id']
                bbox = det['bbox']
                bbox_w = bbox[2] - bbox[0]
                if frame_count % 5 == 0 and bbox_w > 70:
                    plate_recognizer.add_to_queue(frame, obj_id, bbox)

            # E. RENDERING
            current_objects = manager.get_tracks()
            draw_hud(frame, current_objects)

            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("SafeDrive", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_loader.release()
        cv2.destroyAllWindows()


    except Exception as e:
        print(f"Errore critico: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()