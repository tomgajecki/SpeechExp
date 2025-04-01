from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

class BaseWindow(QMainWindow):
    """Base window class with common functionality."""
    
    def __init__(self, eng, title):
        super().__init__()
        self.eng = eng  # Store the engine
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 600)
        self.setFixedSize(800, 600)
        self.center()
        
        # Create central widget and layout
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # Status label
        self.statusLabel = QLabel("Initializing...")
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.statusLabel)
        
        # Progress bar
        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)
        self.layout.addWidget(self.progressBar)
        
        # Apply styles
        self.applyStyles()
    
    def initUI(self):
        """Initialize the user interface."""
        pass  # To be implemented by child classes
    
    def center(self):
        """Center the window on the screen."""
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def applyStyles(self):
        """Apply common styles to the window."""
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #4d4d4d;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 5px;
                border-radius: 4px;
                min-width: 200px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #4d4d4d;
            }
        """) 