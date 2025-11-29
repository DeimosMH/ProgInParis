from PyQt6.QtCore import QObject, pyqtSignal

class EventBus(QObject):
    dataforchart = pyqtSignal()

event_bus = EventBus()
