from abc import ABC, abstractmethod

class StatoVeicolo(ABC):

    @abstractmethod    #Dice: “Ogni sottoclasse deve implementare questo metodo, altrimenti non può essere istanziata.”
    @property
    def color(self):
        pass

    @property           #Trasforma una funzione in una proprietà: cioè lo richiami senza parentesi. Invece di scrivere obj.get_color(), puoi scrivere semplicemente obj.color.
    @abstractmethod    #Dice: “Ogni sottoclasse deve implementare questo metodo, altrimenti non può essere istanziata.”
    def name (self):
        pass

class SafeState(StatoVeicolo):
    @property
    def color(self):
        return (0, 255,0)  # Verde

    @property
    def name(self):
        return "Safe"
    
    class WarningState(StatoVeicolo):
        @property
        def color(self):
            return (0, 255, 255)  # Giallo
        
        @property
        def name(self):
            return "Warning"
        
class DangerState(StatoVeicolo):
    @property
    def color(self):
        return (0, 0, 255)  # Rosso

    @property
    def name(self):
        return "Danger"
    
    
