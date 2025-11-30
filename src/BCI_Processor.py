from pynput.mouse import Controller as dick,Button
from pynput.keyboard import Controller,Key
from src.event_bus import event_bus
from src.text_bus import text_bus
from src.abstractBCIProcessor import abstractBCIProcessor
import time

moveSpeed: int = 50

class BCIProcessor(abstractBCIProcessor):
    """
    A class containing the static methods that perform the physical or 
    computational action associated with a detected BCI event.
    """

    @staticmethod
    def doubleBlink()->None:
        text_bus.textToSend.emit("Blinked twice")
        print("BCIProcessor: Executing Double Blink Action.")
        mouse = dick() 
        mouse.click(Button.left)
        
    
    @staticmethod
    def tripleBlink()->None:
        text_bus.textToSend.emit("Blinked trice")
        print("BCIProcessor: Executing Triple Blink Action.")
        mouse = dick()
        mouse.position=(1920/2,1080/2)
        
    @staticmethod
    def blinkOneEye()->None:
        text_bus.textToSend.emit("Blinked with one eye")
        print("BCIProcessor: Executing Blink Right Action.")
        mouse = dick()
        mouse.click(Button.right)
        
   
        
    @staticmethod
    def lookRight()->None:
        text_bus.textToSend.emit("Looked Right")
        print("BCIProcessor: Executing Look Right Action.")
        mouse = dick()
        mouse.move(moveSpeed,0)

        
    @staticmethod
    def lookLeft()->None:
        text_bus.textToSend.emit("Looked left")
        print("BCIProcessor: Executing Look Left Action.")
        mouse = dick()
        mouse.move(-moveSpeed,0)
        
    @staticmethod
    def lookUp()->None:
        text_bus.textToSend.emit("Looked up")
        print("BCIProcessor: Executing Look Up Action.")
        mouse = dick()
        mouse.move(0,-moveSpeed)
        
    @staticmethod
    def lookDown()->None:
        text_bus.textToSend.emit("Looked down")
        print("BCIProcessor: Executing Look Down Action.")
        mouse = dick()
        mouse.move(0,moveSpeed)
    

    @staticmethod
    def signalData(data):
        event_bus.dataforchart.emit(data)
        
