from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

class CustomMessageBox(QMessageBox):
    """Standardized message box with consistent styling."""
    def __init__(self, parent=None, title="", text="", icon=QMessageBox.Information, detailed_text=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setText(text)
        self.setIcon(icon)
        if detailed_text:
            self.setDetailedText(detailed_text)
            
        # Set consistent styling
        self.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: black;
                font-size: 12px;
                min-width: 300px;
            }
            QPushButton {
                background-color: #2c3e50;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QTextEdit {
                color: black;
                font-size: 12px;
            }
        """) 