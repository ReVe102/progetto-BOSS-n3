import motmetrics as mm
import numpy as np
import os

try:
    np.asfarray
except AttributeError:
    np.asfarray = lambda x: np.asarray(x, dtype=float)
    
class MOTMetricsManager:
    def __init__(self, gt_file_path):
        """
        Inizializza l'accumulatore e carica la Ground Truth (le soluzioni) in memoria.
        """
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.gt_data = self._load_gt(gt_file_path)
        print(f"[METRICHE] Caricata Ground Truth da: {gt_file_path}")

    def _load_gt(self, path):
        """
        Legge il file GT in formato KITTI (separato da spazi).
        Formato KITTI: frame id type truncated occluded alpha x1 y1 x2 y2 ...
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File gt.txt non trovato: {path}")

        gt_dict = {}
        with open(path, 'r') as f:
            for line in f:
                # 1. Usa split() senza argomenti per dividere sugli spazi
                parts = line.strip().split()
                
                # 2. Parsing specifico per KITTI
                frame = int(parts[0])
                obj_id = int(parts[1])
                obj_type = parts[2]  # Es. 'Car', 'Van', 'DontCare'
                
                # Le coordinate in KITTI sono alle posizioni 6, 7, 8, 9
                # Format: left, top, right, bottom
                x1 = float(parts[6])
                y1 = float(parts[7])
                x2 = float(parts[8])
                y2 = float(parts[9])
                
                # Calcoliamo larghezza e altezza
                w = x2 - x1
                h = y2 - y1

                # 3. Filtri opzionali
                # In KITTI, ID -1 è solitamente 'DontCare'. Lo ignoriamo per le metriche.
                if obj_id < 0:
                    continue
                
                # Se vuoi tracciare SOLO le auto, puoi scommentare questo:
                # if obj_type not in ['Car', 'Van', 'Truck']:
                #    continue

                if frame not in gt_dict:
                    gt_dict[frame] = {'ids': [], 'boxes': []}
                
                gt_dict[frame]['ids'].append(obj_id)
                # Salviamo come [x, y, w, h] per motmetrics
                gt_dict[frame]['boxes'].append([x1, y1, w, h])
        
        return gt_dict

    def update(self, frame_number, detected_objects):
        """
        Confronta le detection di YOLO con la Ground Truth per il frame corrente.
        """
        # 1. Recupera la verità per questo frame
        if frame_number not in self.gt_data:
            # Se il video è più lungo del file di verità, o se quel frame non ha dati
            self.acc.update([], [], [])
            return

        gt_ids = self.gt_data[frame_number]['ids']
        gt_boxes = self.gt_data[frame_number]['boxes'] # [x, y, w, h]

        # 2. Prepara le tue detection (Hypothesis)
        pred_ids = []
        pred_boxes = []

        # --- FILTRO CLASSI (Cruciale per YOLO su Dataset KITTI) ---
        # YOLOv8 (COCO) usa questi ID: 2=Auto, 3=Moto, 5=Bus, 7=Camion
        # Se non filtriamo, YOLO proverà a tracciare panchine o semafori 
        # e le metriche crolleranno perché il dataset traccia solo veicoli.
        target_coco_classes = [2, 3, 5, 7] 

        for obj in detected_objects:
            # Estraiamo i dati dall'oggetto (supporta sia dict che TrackedObject)
            if isinstance(obj, dict):
                # Caso raw detection
                x1, y1, x2, y2 = obj['bbox']
                oid = obj['id']
                cls = obj.get('class_id', -1)
            else:
                # Caso oggetto TrackedObject (dal tuo codice)
                x1, y1, x2, y2 = obj.info['bbox']
                oid = obj.id
                cls = obj.info['class_id']
            
            # SOLO se è un veicolo lo passiamo al calcolo metriche
            if cls in target_coco_classes:
                w = x2 - x1
                h = y2 - y1
                pred_ids.append(oid)
                pred_boxes.append([x1, y1, w, h])

        # 3. Calcola Distanze (IoU)
        # motmetrics calcola la matrice dei costi
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

        # 4. Aggiorna Accumulatore
        self.acc.update(gt_ids, pred_ids, distances)

    def print_summary(self):
        """Stampa la tabella finale."""
        mh = mm.metrics.create()
        
        # Le metriche richieste dal tuo progetto
        metrics_list = ['num_frames', 'mota', 'idf1', 'idp', 'idr', 'num_switches']
        
        # Calcolo
        summary = mh.compute(self.acc, metrics=metrics_list, name='Report')
        
        # Formattazione
        strsummary = mm.io.render_summary(
            summary, 
            formatters=mh.formatters, 
            namemap=mm.io.motchallenge_metric_names
        )
        
        print("\n" + "="*60)
        print(" REPORT METRICHE SCIENTIFICHE (py-motmetrics)")
        print("="*60)
        print(strsummary)
        print("="*60 + "\n")