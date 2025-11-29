from pynput.mouse import Controller as dick,Button
from pynput.keyboard import Controller,Key
import time

class BCIProcessor:
    """
    A class containing the static methods that perform the physical or 
    computational action associated with a detected BCI event.
    """
    
    @staticmethod
    def doubleBlink():
        print("BCIProcessor: Executing Double Blink Action.")
    
    @staticmethod
    def tripleBlink():
        print("BCIProcessor: Executing Triple Blink Action.")
        
    @staticmethod
    def blinkRight():
        print("BCIProcessor: Executing Blink Right Action.")
        mouse = dick()
        mouse.click(Button.right)
        
    @staticmethod
    def blinkLeft():
        print("BCIProcessor: Executing Blink Left Action.")
        mouse= dick()
        mouse.click(Button.left)
        
    @staticmethod
    def lookRight():
        print("BCIProcessor: Executing Look Right Action.")
        
    @staticmethod
    def lookLeft():
        print("BCIProcessor: Executing Look Left Action.")
        
    @staticmethod
    def lookUp():

        print("BCIProcessor: Executing Look Up Action.")
        
    @staticmethod
    def lookDown():
        print("BCIProcessor: Executing Look Down Action.")

time.sleep(3)
BCIProcessor.blinkLeft()