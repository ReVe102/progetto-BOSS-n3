from abc import ABC, abstractmethod

# --- 1. INTERFACCIA STATE (L'astrazione) ---
class VehicleState(ABC):
    """
    Classe astratta che definisce come deve comportarsi uno stato.
    Ogni stato deve avere un colore (per il disegno) e un nome.
    """
    @property
    @abstractmethod
    def color(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

# --- 2. STATI CONCRETI (Le implementazioni) ---
class SafeState(VehicleState):
    """Stato: Lontano o non in traiettoria. Colore: Verde."""
    @property
    def color(self):
        return (0, 255, 0)  # Verde
    
    @property
    def name(self):
        return "SAFE"

class WarningState(VehicleState):
    """Stato: Si sta avvicinando o è a media distanza. Colore: Giallo."""
    @property
    def color(self):
        return (0, 255, 255)  # Giallo
    
    @property
    def name(self):
        return "WARNING"

class DangerState(VehicleState):
    """Stato: Vicino e in traiettoria di collisione. Colore: Rosso."""
    @property
    def color(self):
        return (0, 0, 255)  # Rosso
    
    @property
    def name(self):
        return "DANGER"

# 3. CONTEXT (L'oggetto tracciato) 
class TrackedObject:
    """
    Rappresenta un veicolo tracciato. Mantiene il suo Stato corrente.
    """
    def __init__(self, obj_id, initial_info):
        self.id = obj_id
        self.info = initial_info
        self.state = SafeState()  # Stato iniziale di default
        
        # Storico delle aree per capire se si avvicina (non usato in questa versione base ma utile)
        self.area_history = [] 
        self.state_buffer = []

    def update(self, new_info, frame_width, frame_height):
        """
        Aggiorna i dati dell'oggetto e ricalcola lo stato.
        """
        self.info = new_info
        bbox = new_info['bbox']
        
        # Calcoli geometrici
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        video_area = frame_width * frame_height
        area_ratio = area / video_area  # Quanto spazio occupa nel frame (0.0 a 1.0)

        # --- CALCOLO TTC (Comportamentale) ---
        ttc = float('inf')
        if len(self.area_history) > 0:
            # Calcoliamo la media delle ultime aree per stabilizzare il calcolo
            avg_prev_area = sum(self.area_history[-3:]) / len(self.area_history[-3:])
            diff_area = area - avg_prev_area
            
            # Questo ignora le oscillazioni random di YOLO sulle auto ferme a lato
            if diff_area > (area * 0.03): 
                ttc = area / diff_area

        # Stampa i dati TTC nel terminale per ogni auto
        if ttc != float('inf'):
            print(f"[DEBUG] Veicolo ID {self.id}: TTC = {ttc:.2f} frame | Ratio Area = {area_ratio:.4f}")        

        # Aggiornamento storico aree
        self.area_history.append(area)
        if len(self.area_history) > 20: self.area_history.pop(0)
        
        center_x = new_info['center'][0]
        center_y = new_info['center'][1]

        # Più l'auto è in basso (y alto), più la corsia considerata è larga
        horizon_ratio = center_y / frame_height
        lane_width = 0.15 + (horizon_ratio * 0.25)

        # Definizione della "Zona Centrale" (Traiettoria di collisione)
        lane_start = frame_width * (0.5 - lane_width/2)
        lane_end = frame_width * (0.5 + lane_width/2)
        is_in_lane = lane_start < center_x < lane_end

        # --- LOGICA DI TRANSIZIONE ROBUSTA ---
        new_proposed_state = SafeState()

        if is_in_lane:
            if area_ratio > 0.20 or ttc < 3: 
                new_proposed_state = DangerState()
            elif area_ratio > 0.11 or ttc < 20:
                new_proposed_state = WarningState()
        
        # Se l'auto è fuori corsia ma è gigantesca (ci sta tagliando la strada)
        elif area_ratio > 0.50:
            new_proposed_state = WarningState()

        # 4. FILTRO DI STABILITÀ (Anti-Flickering)
        # Memorizziamo la proposta e cambiamo solo se c'è una maggioranza chiara
        self.state_buffer.append(new_proposed_state.name)
        if len(self.state_buffer) > 8: 
            self.state_buffer.pop(0)

        # Cambiamo stato solo se abbiamo almeno 6 conferme su 8 frame
        # Questo rende il sistema solido e non "nervoso"
        current_proposal_count = self.state_buffer.count(new_proposed_state.name)
        if current_proposal_count >= 7:
            self.set_state(new_proposed_state)

    def set_state(self, new_state):
        """Cambia lo stato corrente."""
        if type(self.state) != type(new_state):
            print(f"Veicolo {self.id}: {self.state.name} -> {new_state.name}") # Debug opzionale
            self.state = new_state