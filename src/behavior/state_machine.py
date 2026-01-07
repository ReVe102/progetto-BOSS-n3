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
        Aggiorna i dati dell'oggetto e ricalcola lo stato con transizioni graduali.
        """
        self.info = new_info
        bbox = new_info['bbox']
        
        # Calcoli geometrici
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        video_area = frame_width * frame_height
        area_ratio = area / video_area # Quanto spazio occupa nel frame (0.0 a 1.0)

        # --- CALCOLO TTC (Comportamentale) ---
        ttc = float('inf')
        if len(self.area_history) > 0:
            # Calcoliamo la media delle ultime aree per stabilizzare il calcolo
            avg_prev_area = sum(self.area_history[-5:]) / len(self.area_history[-5:])
            diff_area = area - avg_prev_area
            # Questo ignora le oscillazioni random di YOLO sulle auto ferme a lato
            if diff_area > (area * 0.05): 
                ttc = area / diff_area

        # Stampa i dati TTC nel terminale per ogni auto
        if ttc != float('inf'):
            print(f"[DEBUG] Veicolo ID {self.id}: TTC = {ttc:.1f} frame | Ratio Area = {area_ratio:.3f}")         
        # Aggiornamento storico aree
        self.area_history.append(area)
        if len(self.area_history) > 20: self.area_history.pop(0)
        
        center_x = new_info['center'][0]
        center_y = new_info['center'][1]
        
        # Più l'auto è in basso (y alto), più la corsia considerata è larga
        horizon_ratio = center_y / frame_height
        lane_width = 0.10 + (horizon_ratio * 0.20)
        # Definizione della "Zona Centrale" (Traiettoria di collisione)
        lane_start = frame_width * (0.5 - lane_width/2)
        lane_end = frame_width * (0.5 + lane_width/2)
        is_in_lane = lane_start < center_x < lane_end

        # ============================================================
        # 1. CALCOLO DELLO STATO "GEOMETRICO" PURO (Target)
        # ============================================================
        # Definiamo cosa DOVREBBE essere in base ai numeri attuali
        target_state = SafeState()

        # SOGLIE DIMENSIONALI
        # > 0.14 : DANGER (Molto grande)
        # > 0.09 : WARNING (Medio)
        # < 0.09 : SAFE (Piccolo)

        if area_ratio > 0.14:
            target_state = DangerState()
        
        elif is_in_lane:
            # Se è in corsia, usiamo le soglie intermedie o il TTC
            if ttc < 3:
                target_state = DangerState()
            elif area_ratio > 0.09 or ttc < 15:
                target_state = WarningState()
        
        # Gestione sorpassi vicini (fuori corsia ma grandi)
        elif area_ratio > 0.12: # Soglia leggermente più alta per chi è fuori corsia
             target_state = WarningState()

        # ============================================================
        # 2. LOGICA SEQUENZIALE (Impedire salti bruschi)
        # ============================================================
        # Assegniamo un livello di priorità agli stati per fare matematica
        def get_level(state_obj):
            if state_obj.name == "DANGER": return 2
            if state_obj.name == "WARNING": return 1
            return 0 # SAFE
        
        current_level = get_level(self.state)
        target_level = get_level(target_state)

        # RECOLA: Non si può scendere di 2 gradini in un colpo solo.
        # Se siamo a DANGER (2) e il target è SAFE (0), forziamo WARNING (1).
        final_proposed_state = target_state

        if current_level == 2 and target_level == 0:
            final_proposed_state = WarningState() # Forziamo lo step intermedio
        
        # Nota: Salire di colpo (Safe -> Danger) è permesso per sicurezza!
        # Scendere invece deve essere graduale per estetica e stabilità.

        # ============================================================
        # 3. FILTRO DI STABILITÀ (Buffer)
        # ============================================================
        self.state_buffer.append(final_proposed_state.name)
        if len(self.state_buffer) > 10: 
            self.state_buffer.pop(0)

        # Applichiamo il cambio solo se c'è consistenza nel buffer
        # Abbassiamo la soglia a 6/10 per rendere il passaggio intermedio visibile
        if self.state_buffer.count(final_proposed_state.name) >= 6:
            self.set_state(final_proposed_state)

    def set_state(self, new_state):
        """Cambia lo stato corrente."""
        if type(self.state) != type(new_state):
            print(f"Veicolo {self.id}: {self.state.name} -> {new_state.name}") # Debug opzionale
            self.state = new_state