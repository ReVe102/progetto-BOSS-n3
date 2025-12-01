import cv2
from src.input_ouput.video_facade import VideoInputFacade
from src.processing.detector import ObjectDetector
from src.behavior.state_machine import TrackedObject
import traceback
import time 

def main():
    # CONFIGURAZIONE
    video_path = "assets/video4" # Sostituisci con 0 per la webcam
    # Usiamo il modello "Small" per un buon compromesso precisione/velocità
    model_name = "yolov8s.pt" 
    
    # INIZIALIZZAZIONE MODULI
    try:
        video_loader = VideoInputFacade(video_path)
        # Otteniamo le dimensioni del video per i calcoli di rischio
        video_width, video_height, fps = video_loader.get_video_info()
        detector = ObjectDetector(model_name=model_name)
        
        # NUOVO: MEMORIA DEGLI OGGETTI 
        # Questo dizionario collegherà l'ID (es. 42) all'oggetto TrackedObject
        
        tracked_objects_memory = {} 

        print(f"Avvio sistema... Video: {video_width}x{video_height} a {fps:.1f} FPS")

        while True:
            # MISURAZIONE TEMPO INIZIALE DEL FRAME (PER CALCOLO FPS)
            frame_start_time = time.time()

            frame = video_loader.get_frame()
            if frame is None: break 


            # 1. RILEVAMENTO E TRACKING 
            detections = detector.detect_and_track(frame)
            
            current_frame_ids = set()

            # 2. AGGIORNAMENTO STATI (LOGICA)
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
                tracked_obj.update(det, video_width, video_height, fps)

            # 2.5. GESTIONE OGGETTI PERSI
            # Iteriamo su TUTTA la memoria per trovare gli oggetti che non sono in questo frame
            for obj_id, tracked_obj in list(tracked_objects_memory.items()):
                if obj_id not in current_frame_ids:                    
                    # Se non è presente, incrementiamo il contatore dei frame persi
                    tracked_obj.frames_lost += 1
                    
                    # Logica di Eliminazione
                    # Se è stato perso per più di 15 frame, lo cancelliamo dalla memoria
                    if tracked_obj.frames_lost > 15: 
                        del tracked_objects_memory[obj_id]
                        print(f"Eliminato Veicolo {obj_id} per perdita di traccia.")

            # --- CALCOLO DEL RISCHIO AGGREGATO ---
            RISK_LEVELS = {'DANGER': 3, 'WARNING': 2, 'SAFE': 1}
            max_risk = 'SAFE'
            max_risk_level = 1
            
            for obj_id, tracked_obj in tracked_objects_memory.items():
                if obj_id in current_frame_ids:
                    current_level = RISK_LEVELS.get(tracked_obj.state.name, 1)
                    if current_level > max_risk_level:
                        max_risk_level = current_level
                        max_risk = tracked_obj.state.name

            # 3. VISUALIZZAZIONE (DISEGNO)
            # Disegniamo solo gli oggetti presenti in QUESTO frame
            for obj_id in current_frame_ids:
                tracked_obj = tracked_objects_memory[obj_id]
                x1, y1, x2, y2 = tracked_obj.info['bbox']
                
                # PRENDIAMO I DATI DAL CONTEXT
                color = tracked_obj.state.color 
                state_name = tracked_obj.state.name
                
                # Recupero delle metriche calcolate
                ttc = tracked_obj.info.get('TTC', float('inf'))
                avg_v_proxy = tracked_obj.info.get('avg_velocity_proxy', 0)
                
                # Formattazione TTC e Velocità per la visualizzazione
                ttc_str = f"TTC: {ttc:.2f} s" if ttc < float('inf') else "TTC: Inf"
                v_str = f"V. PROXY: {avg_v_proxy:.0f}"
                
                # Disegna il rettangolo con il colore dello stato
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # --- Linea 1: ID e STATO ---
                label_state = f"ID:{obj_id} [{state_name}]"
                cv2.putText(frame, label_state, (x1, y1 - 25), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                 
                # --- Linea 2: TTC ---
                cv2.putText(frame, ttc_str, (x1, y1 - 8), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # --- Linea 3: Velocità Proxy ---
                # Per chiarezza, la mettiamo sopra il box
                cv2.putText(frame, v_str, (x1, y1 + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- MISURAZIONE TEMPO FINALE DEL FRAME ---
            frame_end_time = time.time()
            fps_actual = 1 / (frame_end_time - frame_start_time)

            # --- DASHBOARD DI SISTEMA (IN ALTO A SINISTRA) ---
            global_color = (0, 255, 0) # Verde
            if max_risk == 'WARNING': global_color = (0, 255, 255) # Giallo
            if max_risk == 'DANGER': global_color = (0, 0, 255) # Rosso

            # Linea 1: Rischio Aggregato (colore globale)
            cv2.putText(frame, f"RISCHIO AGGREGATO: {max_risk}", (10, 30), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, global_color, 2)

            # Linea 2: Statistiche di Sistema (bianco)
            cv2.putText(frame, f"FPS: {fps_actual:.1f} | Tracciati: {len(current_frame_ids)}", (10, 60), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # FINESTRA DI OUTPUT 
            # Ridimensioniamo per fluidità se il video è grande
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("SafeDrive - State Machine Test", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        video_loader.release()
        
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

        traceback.print_exc() # Stampa l'errore completo per debug

if __name__ == "__main__":
    main()