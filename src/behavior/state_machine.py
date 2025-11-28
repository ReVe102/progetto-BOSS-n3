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
        self.frames_seen = 1
        self.frames_lost = 0 # Contatore per la perdita di traccia
        self.area_history = [] # Storico delle aree per capire se si avvicina (non usato in questa versione base ma utile)
        self.previous_info = None  # Info del frame precedente
        self.velocity_history = [] # Cronologia delle velocità (utile per media)
        self.distance_history = [] # Cronologia delle distanze (utile per Time-to-Collision)

    def update(self, new_info, frame_width, frame_height, fps):
        """
        Aggiorna i dati dell'oggetto e ricalcola lo stato.
        """
        self.info = new_info
        bbox = new_info['bbox']

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        video_area = frame_width * frame_height
        area_ratio = max(area / video_area, 1e-6)

        center_x = new_info['center'][0]
        lane_start = frame_width * 0.3
        lane_end = frame_width * 0.7
        is_in_lane = lane_start < center_x < lane_end

        avg_velocity_proxy = 0
        ttc = float('inf')

        # --- CALCOLO VELOCITÀ E DISTANZA ---
        if self.previous_info is not None:
            
            # Calcolo della Distanza        
            current_area = area
            prev_bbox = self.previous_info['bbox']
            previous_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
            
            # Calcolo della Variazione dell'Area 
            area_change = current_area - previous_area
            
            velocity_proxy = area_change
            self.velocity_history.append(velocity_proxy)

            # Le 5 misurazioni per la velocità media
            self.velocity_history = self.velocity_history[-5:] 

            # Calcolo della V_media (media della variazione di area negli ultimi 5 frame)
            avg_velocity_proxy = sum(self.velocity_history) / len(self.velocity_history) if self.velocity_history else 0
            
            center_x = new_info['center'][0]
            
            # La Distanza (proxy) è l'inverso dell'area corrente (1 / area_ratio)
            DISTANCE_PROXY = 1 / area_ratio

            # La Velocità Relativa (proxy) è la variazione media di area.
            VELOCITY_PROXY = self.info.get('avg_velocity_proxy', 0)

            ttc = float('inf') # Inizializza a infinito (nessun rischio)

            # Calcola TTC solo se l'oggetto si sta avvicinando (velocità positiva)
            if VELOCITY_PROXY > 0.0: 
                # Assumiamo che la velocità sia in "unità di area/frame"
                ttc_in_frames = DISTANCE_PROXY / VELOCITY_PROXY 
                # Converti i frame in secondi
                ttc = ttc_in_frames / fps 
            
            self.info['TTC'] = ttc
            self.info['avg_velocity_proxy'] = avg_velocity_proxy
            self.previous_info = new_info


        # --- LOGICA DI TRANSIZIONE DI STATO ---
        # Questa è la parte "intelligente" che decide il rischio
        
        # Condizione: L'oggetto si sta avvicinando velocemente (alta variazione di area)
        IS_APPROACHING_FAST = self.info.get('avg_velocity_proxy', 0) > 100 
        
        # Condizione: L'oggetto è molto vicino (grande area)
        IS_CLOSE = area_ratio > 0.10
        
        # Condizione: L'oggetto è in traiettoria
        IS_IN_LANE = lane_start < center_x < lane_end

        TTC_CRITICO = 3.0  # Meno di 3 secondi è altissimo rischio
        TTC_ATTENZIONE = 6.0 # Meno di 6 secondi richiede attenzione
        
        
        if self.info['TTC'] < TTC_CRITICO and IS_IN_LANE:
            # Rischio massimo: Collisione prevista entro 3 secondi ed è in traiettoria!
            self.set_state(DangerState())
        
        elif IS_IN_LANE and IS_APPROACHING_FAST and IS_CLOSE: 
            # Condizione 1: In corsia, si avvicina velocemente ed è vicino
            self.set_state(DangerState())

        elif self.info['TTC'] < TTC_ATTENZIONE and IS_IN_LANE:
            # Alto rischio: Collisione prevista entro 6 secondi
            self.set_state(WarningState())    
                   
        elif IS_IN_LANE and IS_APPROACHING_FAST:
            # Condizione 2: In corsia e si sta avvicinando velocemente
            self.set_state(WarningState())
            
        elif IS_CLOSE and not IS_IN_LANE:
            # Condizione 3: È molto vicino, ma non in corsia (es. ci sta sorpassando o è a lato)
            self.set_state(WarningState())
            
        elif IS_IN_LANE and area_ratio > 0.05:
            # Condizione 4: In corsia, non veloce, ma a media distanza
            self.set_state(WarningState())
            
        else:
            # Condizione 5: Tutto è sicuro
            self.set_state(SafeState())

    def set_state(self, new_state):
        """Cambia lo stato corrente."""
        if type(self.state) != type(new_state):
            print(f"Veicolo {self.id}: {self.state.name} -> {new_state.name}") # Debug opzionale
            self.state = new_state