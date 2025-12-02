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
        
        center_x = new_info['center'][0]
        
        # Definizione della "Zona Centrale" (Traiettoria di collisione)
        # Consideriamo il 40% centrale dello schermo come la nostra corsia
        lane_start = frame_width * 0.3
        lane_end = frame_width * 0.7
        is_in_lane = lane_start < center_x < lane_end

        # --- LOGICA DI TRANSIZIONE DI STATO ---
        # Questa è la parte "intelligente" che decide il rischio
        
        if is_in_lane and area_ratio > 0.15: 
            # Se è in corsia e occupa più del 15% dello schermo -> PERICOLO
            self.set_state(DangerState())
            
        elif is_in_lane and area_ratio > 0.05:
            # Se è in corsia e occupa più del 5% -> ATTENZIONE
            self.set_state(WarningState())
            
        elif area_ratio > 0.20:
             # Se è enorme ma non è in corsia (es. auto che sorpassa vicina) -> ATTENZIONE
            self.set_state(WarningState())
            
        else:
            # Altrimenti è sicuro
            self.set_state(SafeState())

    def set_state(self, new_state):
        """Cambia lo stato corrente."""
        if type(self.state) != type(new_state):
            print(f"Veicolo {self.id}: {self.state.name} -> {new_state.name}") # Debug opzionale
            self.state = new_state