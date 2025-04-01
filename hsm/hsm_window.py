from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QComboBox, QVBoxLayout, QWidget, QApplication
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os

class HSMTestWindow(QMainWindow):
    """HSM test window implementation."""
    
    # Signal to notify when window is closing
    window_closing = pyqtSignal()
    
    def __init__(self, eng, name, surname, id_val, map_data=None, map_number=None, map_side=None):
        super().__init__()
        self.eng = eng
        self.name = name
        self.surname = surname
        self.id_val = id_val
        self.map_data = map_data
        self.map_number = map_number or "Default"  # Store map number with default value
        self.map_side = map_side or "Default"  # Store map side with default value
        
        # Set audio folder
        self.audio_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hsm', 'audio')
        
        # Add MATLAB functions directory to path
        matlab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'matlab')
        self.eng.addpath(matlab_path, nargout=0)
        
        # Set window properties
        self.setFixedSize(800, 600)
        self.center()
        
        # Create central widget and layout
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # Status label
        self.statusLabel = QLabel("Ready to stream")
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setStyleSheet("color: white;")
        self.layout.addWidget(self.statusLabel)
        
        # Initialize UI
        self.initUI()
        
        # Initialize stream server status
        self.stream_server_running = True  # Server is already running from MatlabEngineLoader
    
    def center(self):
        """Center the window on the screen."""
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def initUI(self):
        """Initialize the user interface."""
        
        infoLabel = QLabel(f"Subject ID: {self.id_val} | Map: {self.map_number} | Side: {self.map_side}")
        infoLabel.setAlignment(Qt.AlignCenter)
        infoLabel.setStyleSheet("color: white; font-size: 14px; padding: 10px; background-color: #2a2a2a; border-radius: 4px;")
        self.layout.addWidget(infoLabel)
        
        # Separator
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #5e5e5e;")
        self.layout.addWidget(separator)
        
        # Refresh button
        refreshButton = QPushButton("Refresh Files")
        refreshButton.clicked.connect(self.refreshAudioFiles)
        self.layout.addWidget(refreshButton)
        
        # Audio file selection
        audioFilesLabel = QLabel("Select Audio Set:")
        audioFilesLabel.setStyleSheet("color: white;")
        self.layout.addWidget(audioFilesLabel)
        
        self.audioFileCombo = QComboBox()
        self.layout.addWidget(self.audioFileCombo)
        
        # Streaming button
        self.streamButton = QPushButton("Play Audio")
        self.streamButton.clicked.connect(self.streamAudio)
        self.layout.addWidget(self.streamButton)
        
        # Go Back button
        backButton = QPushButton("Go Back")
        backButton.clicked.connect(self.goBack)
        self.layout.addWidget(backButton)
        
        # Load initial audio files
        self.refreshAudioFiles()
        
        # Apply styles
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2d6da3;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 5px;
                border-radius: 4px;
                min-width: 200px;
            }
        """)
    
    def refreshAudioFiles(self):
        """Refresh the list of audio files from the audio folder."""
        try:
            self.audioFileCombo.clear()
            
            if os.path.exists(self.audio_folder):
                wav_files = [f for f in os.listdir(self.audio_folder) 
                            if f.lower().endswith('.wav')]
                
                
                for filename in sorted(wav_files):
                    self.audioFileCombo.addItem(filename)
            
            if self.audioFileCombo.count() == 0:
                self.statusLabel.setText("No HSM files found in folder. Using defaults.")
                self.statusLabel.setStyleSheet("color: #ffc107;")
            else:
                self.statusLabel.setText(f"Found {self.audioFileCombo.count()} HSM files.")
                self.statusLabel.setStyleSheet("color: #28a745;")
                
        except Exception as e:
            default_files = ["choice.wav", "test1.wav", "test2.wav"]
            self.audioFileCombo.clear()
            self.audioFileCombo.addItems(default_files)
            self.statusLabel.setText(f"Error refreshing audio files: {str(e)}")
            self.statusLabel.setStyleSheet("color: #dc3545;")
            print(f"Error refreshing audio files: {e}")
    
    def streamAudio(self):
        """Stream the selected audio file."""
        if not hasattr(self, 'stream_server_running') or not self.stream_server_running:
            self.statusLabel.setText("Stream server not running. Cannot play audio.")
            self.statusLabel.setStyleSheet("color: #dc3545;")
            return
        
        selected_file = self.audioFileCombo.currentText()
        audio_path = f"{self.audio_folder}/{selected_file}"
        
        try:
            self.statusLabel.setText(f"Playing {selected_file}...")
            self.statusLabel.setStyleSheet("color: #ffffff;")
            
            self.streamButton.setEnabled(False)
            
            if self.map_data:
                self.eng.stream(self.map_data, audio_path, nargout=0)
            else:
                self.eng.eval(f"p = ACE_map(); stream(p, '{audio_path}')", nargout=0)
            
            self.statusLabel.setText(f"Successfully played {selected_file}")
            self.statusLabel.setStyleSheet("color: #28a745;")
        except Exception as e:
            self.statusLabel.setText(f"Error playing audio: {str(e)}")
            self.statusLabel.setStyleSheet("color: #dc3545;")
            print(f"Error playing audio: {e}")
        finally:
            self.streamButton.setEnabled(True)
    
    def goBack(self):
        """Go back to the main window."""
        self.hide()  # Hide this window
        # Find and show the main window
        for widget in QApplication.topLevelWidgets():
            if widget.objectName() == "mainWindow":
                widget.show()
                break
    
    def closeEvent(self, event):
        """Handle window close event."""
        print("HSM test window closeEvent called")
        # Emit the window_closing signal
        self.window_closing.emit()
        print("window_closing signal emitted")
        event.accept() 