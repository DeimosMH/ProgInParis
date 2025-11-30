from PySide6.QtCore import QObject, QThread, Signal
import time
from src.Messenger import Messenger
class Worker(QObject):
    
    finished = Signal()       # When thread ends
    
    def run(self):
        message = Messenger()
        time.sleep(3)
        message.set_action("double_blink")
        message.process()
        time.sleep(3)
        message.set_action("blink_right")
        message.process()
        time.sleep(3)
        message.set_action("look_up")
        message.process()
        time.sleep(3)
        message.set_action("double_blink")
        message.set_action("look_right")
        message.process()
        
        
            
    