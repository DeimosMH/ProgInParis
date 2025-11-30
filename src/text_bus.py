from PySide6.QtCore import QObject, Signal

class TextBus(QObject):
    textToSend = Signal(str)

text_bus = TextBus()