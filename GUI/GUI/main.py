import sys
import os
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QThread, Signal, QObject

# --- CZEŚĆ 1: INNA CZĘŚĆ PROGRAMU (np. Odczyt danych) ---
# To działa w tle i nie blokuje okienka
class DataWorker(QThread):
    # Definiujemy sygnał, który będzie niósł tekst (str)
    nowe_dane_signal = Signal(str)

    def run(self):
        licznik = 0
        while True:
            # Symulacja pracy (np. odczyt z USB/CAN)
            time.sleep(1) 
            licznik += 1
            
            # Generujemy wiadomość
            wiadomosc = f"Odczytano ramkę danych nr: {licznik}"
            
            # WYSYŁAMY SYGNAŁ do GUI
            self.nowe_dane_signal.emit(wiadomosc)

# --- CZEŚĆ 2: KONTROLER (GUI) ---
class StiCanController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = self.load_ui()
        
        # Tworzymy instancję "innej części programu"
        self.worker = DataWorker()
        
        # --- KLUCZOWY MOMENT: ŁĄCZENIE ---
        # Mówimy: "Jak worker wyśle sygnał 'nowe_dane_signal', uruchom moją funkcję 'changeText'"
        self.worker.nowe_dane_signal.connect(self.changeText)
        
        # Uruchamiamy wątek w tle
        self.worker.start()

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
        html_text = f"<h1>Nowe Dane:</h1><p style='color: blue; font-size: 20px'>{text}</p>"
        
        if hasattr(self.window, 'textBrowser'):
            self.window.textBrowser.setHtml(html_text)
        else:
            print(f"GUI: {text}")

    def run(self):
        self.window.show()
        sys.exit(self.app.exec())

if __name__ == "__main__":
    controller = StiCanController()
    controller.run()