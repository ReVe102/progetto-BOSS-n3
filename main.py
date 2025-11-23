import cv2
from src.input_ouput.video_facade import VideoInputFacade
from src.processing.detector import ObjectDetector
# NUOVO IMPORT: Importiamo la classe che gestisce lo Stato
from src.behavior.state_machine import TrackedObject

def main():
    # --- CONFIGURAZIONE ---
    video_path = "assets/video4.mp4" # Sostituisci con 0 per la webcam
    # Usiamo il modello "Small" per un buon compromesso precisione/velocità
    model_name = "yolov8s.pt" 
    
    # --- INIZIALIZZAZIONE MODULI ---
    try:
        video_loader = VideoInputFacade(video_path)
        # Otteniamo le dimensioni del video per i calcoli di rischio
        video_width, video_height, fps = video_loader.get_video_info()
        
        detector = ObjectDetector(model_name=model_name)
        
        # --- NUOVO: MEMORIA DEGLI OGGETTI ---
        # Questo dizionario collegherà l'ID (es. 42) all'oggetto TrackedObject
        
        tracked_objects_memory = {} 

        print(f"Avvio sistema... Video: {video_width}x{video_height} a {fps:.1f} FPS")

        while True:
            frame = video_loader.get_frame()
            if frame is None: break 

            # --- 1. RILEVAMENTO E TRACKING (AI) ---
            detections = detector.detect_and_track(frame)
            
            current_frame_ids = set()

            # --- 2. AGGIORNAMENTO STATI (LOGICA) ---
            for det in detections:
                obj_id = det['id']
                current_frame_ids.add(obj_id)

                # Se è un oggetto nuovo, lo creiamo
                if obj_id not in tracked_objects_memory:
                    tracked_objects_memory[obj_id] = TrackedObject(obj_id, det)
                
                # Recuperiamo l'oggetto dalla memoria
                tracked_obj = tracked_objects_memory[obj_id]
                
                # AGGIORNIAMO LO STATO
                # L'oggetto ricalcola se è Safe, Warning o Danger
                tracked_obj.update(det, video_width, video_height)

            # 3. VISUALIZZAZIONE (DISEGNO)
            # Disegniamo solo gli oggetti presenti in QUESTO frame
            for obj_id in current_frame_ids:
                tracked_obj = tracked_objects_memory[obj_id]
                x1, y1, x2, y2 = tracked_obj.info['bbox']
                
                # PRENDIAMO IL COLORE DALLO STATO CORRENTE
                # Safe=Verde, Warning=Giallo, Danger=Rosso
                color = tracked_obj.state.color 
                state_name = tracked_obj.state.name
                
                # Disegna il rettangolo con il colore dello stato
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Scrivi ID e STATO sopra il box
                label = f"ID:{obj_id} [{state_name}]"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # FINESTRA DI OUTPUT 
            # Ridimensioniamo per fluidità se il video è grande
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("SafeDrive - State Machine Test", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        video_loader.release()
        
    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        import traceback
        traceback.print_exc() # Stampa l'errore completo per debug

if __name__ == "__main__":
    main()