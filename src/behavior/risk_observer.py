from abc import ABC, abstractmethod
# Assicurati che l'import sia corretto in base alla tua struttura
from src.behavior.state_machine import TrackedObject, DangerState
from src.behavior.state_machine import SafeState, WarningState

class Observer(ABC):
    @abstractmethod
    def update(self, event_type, track_id, message=""):
        pass

class ConsoleAlertObserver(Observer):
    def update(self, event_type, track_id, message=""):
        if event_type == "DANGER":
            # Codice ANSI per testo ROSSO in console
            print(f"\033[91m[ALLARME] Veicolo {track_id}: {message}\033[0m")  
        elif event_type == "NEW_TRACK":
            print(f"[INFO] Nuova traccia: {track_id}")
        elif event_type == "LOST_TRACK":
            print(f"[INFO] Perso contatto: {track_id}")

class TrackManager:
    """
    SOGGETTO (Subject). Gestisce gli oggetti e notifica gli Observer.
    """
    def __init__(self):
        self.observers = [] #Lista di chi sta ascoltando (es. la Console)
        self.tracks = {} # Memoria delle auto (Dizionario ID -> Oggetto)

    def attach(self, observer):
        self.observers.append(observer) # Aggiunge un nuovo ascoltatore alla lista

    def notify(self, event_type, track_id, message=""):   #Quando succede qualcosa, il Manager non fa print(). Chiama il metodo notify. Questo metodo dice: "Per tutti quelli che mi stanno ascoltando (observers), ecco l'aggiornamento!"
        for observer in self.observers:
            observer.update(event_type, track_id, message)

    def update_tracks(self, detections, frame_w, frame_h):
        active_ids = []

        for det in detections:
            obj_id = det["id"]           
            active_ids.append(obj_id)   #prende id e lo mette nella lista degli attivi

            # 1. È una NUOVA traccia?
            if obj_id not in self.tracks:
                # l'oggetto new_obj che contiene tutta la logica del file state_machine.py
                new_obj = TrackedObject(obj_id, det) # Crea nuovo oggetto
                new_obj.update(det, frame_w, frame_h) # aggiunge l'oggetto
                
                self.tracks[obj_id] = new_obj # Memorizza la traccia
                 #Notifica tutti gli observer che c'è una nuova traccia
                self.notify("NEW_TRACK", obj_id)
            else:
                # 2. Aggiorna traccia ESISTENTE
                current_obj = self.tracks[obj_id]
                old_state_name = current_obj.state.name
                
                 #Qui il flusso di esecuzione SALTA dal file risk_observer.py al file state_machine.py. Dentro state_machine.py, il metodo update fa i calcoli matematici (Area, Centro). Sempre dentro state_machine.py, l'oggetto decide se cambiare il suo stato interno (es. self.state = DangerState()). Finito il calcolo, il flusso torna al Manager.
                current_obj.update(det, frame_w, frame_h)
                
                new_state_name = current_obj.state.name # Il Manager sbircia dentro l'oggetto per vedere lo stato corrente

                # 3. Controllo cambio stato -> PERICOLO
                # Se passa a DANGER e prima non lo era, notifica!
                if new_state_name == "DANGER" and old_state_name != "DANGER":
                    self.notify("DANGER", obj_id, "COLLISIONE IMMINENTE!")

        # 4. Gestione tracce PERSE
        current_track_ids = list(self.tracks.keys())
        for track_id in current_track_ids:
            if track_id not in active_ids:
                del self.tracks[track_id]
                self.notify("LOST_TRACK", track_id)
                
    def get_tracks(self):
        return self.tracks.values()