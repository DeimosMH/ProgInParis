from PySide6.QtCore import QObject, Signal

class EventBus(QObject):
    dataforchart = Signal()

event_bus = EventBus()

