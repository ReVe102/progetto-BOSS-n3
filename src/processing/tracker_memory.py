import cv2
import numpy as np

class VisualMemory:
    """
    Implementa la logica TOOCM:
    1. Aggiornamento Dinamico: Memorizza sempre l'ultima texture vista.
    2. Recupero Storico: Cerca corrispondenze basate su posizione e colore precedente.
    """
    def __init__(self):
        # Struttura: { id: {'hist': istogramma, 'center': (x,y), 'frames_lost': 0} }
        self.history = {}
        
        # PARAMETRI DI RECUPERO (Vanishing Feature Recovery)
        # Se l'oggetto si sposta di max 150px mentre è "perso", lo consideriamo lo stesso.
        self.max_distance = 150 
        # Soglia somiglianza colore (0.0 diverso, 1.0 identico).
        self.color_threshold = 0.50 
        # Quanti frame ricordiamo un oggetto "svanito" (Memory persistence)
        self.max_frames_to_remember = 60 

    def _get_color_hist(self, crop):
        """Estrae la 'texture' sotto forma di istogramma colore."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Usiamo Hue e Saturation per essere robusti alla luce
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def update_memory(self, obj_id, crop, center):
        """
        PRINCIPIO 'DYNAMIC UPDATE'[cite: 14]:
        Aggiorniamo costantemente la rappresentazione dell'oggetto.
        Se l'auto gira o cambia luce, la memoria si adatta al nuovo aspetto.
        """
        if crop.size == 0: return
        
        hist = self._get_color_hist(crop)
        self.history[obj_id] = {
            'hist': hist,            # La "texture" corrente
            'center': center,        # La posizione corrente
            'frames_lost': 0         # È visibile, quindi 0 persi
        }

    def increment_lost_counters(self):
        """Invecchia i ricordi (simula il passare del tempo t)."""
        to_delete = []
        for obj_id in self.history:
            self.history[obj_id]['frames_lost'] += 1
            # Se passa troppo tempo, dimentichiamo l'oggetto
            if self.history[obj_id]['frames_lost'] > self.max_frames_to_remember:
                to_delete.append(obj_id)
        
        for obj_id in to_delete:
            del self.history[obj_id]

    def find_match(self, new_crop, new_center):
        """
        PRINCIPIO 'VANISHING FEATURE RECOVERY'[cite: 447]:
        Cerca tra gli oggetti persi (frames_lost > 0) quello più simile
        per posizione e texture.
        """
        if new_crop.size == 0: return None

        new_hist = self._get_color_hist(new_crop)
        best_id = None
        best_score = 0 

        for old_id, data in self.history.items():
            # Consideriamo solo oggetti che YOLO ha perso (frames_lost >= 1)
            # Se frames_lost è 0, YOLO lo sta già tracciando, non serve intervenire.
            if data['frames_lost'] < 1:
                continue

            # 1. Confronto Posizione (Spostamento nel tempo)
            old_center = data['center']
            dist = np.linalg.norm(np.array(old_center) - np.array(new_center))
            
            if dist > self.max_distance:
                continue 

            # 2. Confronto Texture (Istogramma)
            color_sim = cv2.compareHist(data['hist'], new_hist, cv2.HISTCMP_CORREL)
            
            if color_sim < self.color_threshold:
                continue 

            # Punteggio combinato
            score = color_sim + (1 - (dist / self.max_distance))

            if score > best_score:
                best_score = score
                best_id = old_id

        if best_id is not None:
            print(f"✅ RECOVERY: ID {best_id} recuperato dalla memoria (Score: {best_score:.2f})")
            return best_id

        return None