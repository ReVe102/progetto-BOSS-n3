from ultralytics import YOLO
import cv2


class ObjectDetector:
    """
    Questa classe gestisce il modello AI (YOLO).
    Si occupa di rilevare oggetti e, soprattutto, di TRACCIARLI (dare loro un ID).
    """

    def __init__(self, model_name="yolov8s.pt"):
        """
        Carica il modello YOLO.
        La prima volta che lo lanci, scaricherà automaticamente i pesi da internet.
        """
        print(f"Caricamento modello {model_name}...")
        self.model = YOLO(model_name)
        # Classi che ci interessano (per escludere piante, semafori, ecc se vogliamo)
        # Nel modello standard COCO: 0=persona, 2=auto, 3=moto, 5=bus, 7=camion
        self.target_classes = [0, 2, 3, 5, 7] 

    def detect_and_track(self, frame):
        """
        Esegue il tracking sul frame corrente.
        
        :param frame: L'immagine attuale dal video.
        :return: Una lista di dizionari contenenti: bbox, id, classe, confidenza.
        """
        
        # persist=True è FONDAMENTALE: dice a YOLO di ricordare gli oggetti del frame precedente
        results = self.model.track(source=frame, conf=0.2, iou = 0.65, persist=True, tracker="botsort.yaml", imgsz=640)
        detected_objects = [] 
        
        detected_objects = [] 

        # YOLO può restituire più risultati, prendiamo il primo (il nostro frame)
        result = results[0]
        
        # Se non ci sono oggetti o non ha assegnato ID (succede nei primi frame), saltiamo
        if result.boxes is None or result.boxes.id is None:
            return []

        # Estraiamo i dati grezzi (coordinate, id, classi)
        boxes = result.boxes.xyxy.cpu().numpy()  # Coordinate: x1, y1, x2, y2
        track_ids = result.boxes.id.int().cpu().numpy() # Gli ID univoci (es. 1, 2, 3...)
        class_ids = result.boxes.cls.int().cpu().numpy() # Che cos'è? (auto, persona...)
        
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            # Filtriamo solo le classi che ci interessano (veicoli e persone)
            if class_id in self.target_classes:
                x1, y1, x2, y2 = box
                
                # Creiamo un pacchetto pulito di dati per ogni oggetto
                obj_data = {
                    "id": track_id,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": class_id,
                    "center": (int((x1+x2)/2), int((y1+y2)/2)) # Calcoliamo il centro
                }
                detected_objects.append(obj_data)
                
        return detected_objects