import sys
import os
import time
from src.worker import Worker
from src.event_bus import event_bus
from src.text_bus import text_bus
from PySide6.QtWidgets import QApplication,QPushButton
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QThread, Signal, QObject

class StiCanController:
    def __init__(self):
        
        self.app = QApplication(sys.argv)
        self.window = self.load_ui()
        
        # Tworzymy instancję "innej części programu"
        event_bus.dataforchart.connect(self.drawPlot)
        text_bus.textToSend.connect(self.changeText)
        self.startButton = self.window.findChild(QPushButton, "startButton")
        self.startButton.clicked.connect(self.startThread)
        self.Thread = QThread()
        self.worker=Worker()

    def load_ui(self):
        basedir = os.path.dirname(os.path.abspath(__file__))
        ui_file_name = os.path.join(basedir, "form.ui")
        ui_file = QFile(ui_file_name)

        if not ui_file.open(QIODevice.ReadOnly):
            print(f"BŁĄD: Nie znaleziono {ui_file_name}")
            sys.exit(-1)

        loader = QUiLoader()
        window = loader.load(ui_file)
        ui_file.close()
        return window

    # Ta funkcja zostanie wywołana automatycznie przez sygnał
    def changeText(self, text: str):
        print("changing text")
        html_text = f"<h1>Newest action:</h1><p style='color: blue; font-size: 20px'>{text}</p>"
        
        if hasattr(self.window, 'textBrowser'):
            self.window.textBrowser.setHtml(html_text)
            

        else:
            print(f"GUI: self.window.textBrowser {text}")

    def run(self):
        self.window.show()
        self.changeText("NIGGER")
        sys.exit(self.app.exec())
    
    def drawPlot(self):
        print("to do")
        #counter so it doesnt draw every time

    def startThread(self):
        self.Thread.started.connect(self.worker.run)  
        self.worker.moveToThread(self.Thread)
        self.Thread.start()


controller = StiCanController()
controller.run()