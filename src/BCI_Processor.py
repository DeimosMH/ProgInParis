from pynput.mouse import Controller as dick,Button
from pynput.keyboard import Controller,Key
from src.event_bus import event_bus
from src.text_bus import text_bus
import time

class BCIProcessor:
    """
    A class containing the static methods that perform the physical or 
    computational action associated with a detected BCI event.
    """
    
    @staticmethod
    def doubleBlink():
        text_bus.textToSend.emit("Blinked twice")
        print("BCIProcessor: Executing Double Blink Action.")
    
    @staticmethod
    def tripleBlink():
        text_bus.textToSend.emit("Blinked trice")
        print("BCIProcessor: Executing Triple Blink Action.")
        
    @staticmethod
    def blinkRight():
        text_bus.textToSend.emit("Blinked with right eye")
        print("BCIProcessor: Executing Blink Right Action.")
        mouse = dick()
        mouse.click(Button.right)
        
    @staticmethod
    def blinkLeft():
        text_bus.textToSend.emit("Blinked with left eye")
        print("BCIProcessor: Executing Blink Left Action.")
        mouse= dick()
        mouse.click(Button.left)
        
    @staticmethod
    def lookRight():
        text_bus.textToSend.emit("Looked Right")
        print("BCIProcessor: Executing Look Right Action.")
        
    @staticmethod
    def lookLeft():
        text_bus.textToSend.emit("Looked left")
        print("BCIProcessor: Executing Look Left Action.")
        
    @staticmethod
    def lookUp():
        text_bus.textToSend.emit("Looked up")

        print("BCIProcessor: Executing Look Up Action.")
        
    @staticmethod
    def lookDown():
        text_bus.textToSend.emit("Looked down")
        print("BCIProcessor: Executing Look Down Action.")
    

    @staticmethod
    def signalData(data):
        event_bus.dataforchart.emit(data)
        
