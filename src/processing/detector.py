from ultralytics import YOLO
import cv2
from src.processing.tracker_memory import VisualMemory


class ObjectDetector:
    def __init__(self, model_name="yolov8s.pt"):
        print(f"Caricamento modello {model_name}...")
        self.model = YOLO(model_name)
        self.target_classes = [0, 2, 3, 5, 7]
        
        # Inizializza la memoria dinamica
        self.memory = VisualMemory() 
        
        # Set per evitare conflitti ID nello stesso frame
        self.active_ids_in_frame = set()

    def detect_and_track(self, frame):
       
        self.memory.increment_lost_counters()

        # Tracking YOLO base
        results = self.model.track(source=frame, conf=0.25, iou=0.5, persist=True, tracker="botsort.yaml", imgsz=640, verbose=False)
        
        detected_objects = [] 
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        class_ids = result.boxes.cls.int().cpu().numpy()
        
        h, w, _ = frame.shape
        # Reset ID attivi per questo frame
        self.active_ids_in_frame = set(track_ids)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if class_id in self.target_classes:
                x1, y1, x2, y2 = map(int, box)
                
                # Calcolo Centro
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                current_center = (center_x, center_y)

                final_id = track_id
                
                # Ritaglio Texture Corrente
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if crop.size > 0:
                    # --- LOGICA TOOCM (Adattamento e Recupero) ---
                    
                    # 1. Se YOLO assegna un ID NUOVO, verifichiamo se è un "Vanish Feature" recuperabile
                    # Controlliamo se matcha con un oggetto perso recentemente
                    matched_id = self.memory.find_match(crop, current_center)
                    
                    if matched_id is not None:
                        # Se troviamo un match nella storia E non c'è conflitto nel frame attuale
                        if matched_id not in self.active_ids_in_frame:
                            final_id = matched_id
                            # Trucco: Aggiungiamo il vecchio ID ai "presenti" per evitare che altri lo usino
                            self.active_ids_in_frame.add(final_id)
                    
                    # 2. DYNAMIC UPDATE[cite: 14]:
                    # Indipendentemente se l'ID è vecchio o recuperato, AGGIORNIAMO la memoria.
                    # Questo permette al sistema di "imparare" il nuovo aspetto dell'auto (es. se ha girato).
                    self.memory.update_memory(final_id, crop, current_center)

                obj_data = {
                    "id": final_id,
                    "bbox": (x1, y1, x2, y2),
                    "class_id": class_id,
                    "center": current_center
                }
                detected_objects.append(obj_data)
                
        return detected_objects