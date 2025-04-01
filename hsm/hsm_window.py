from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QComboBox, QVBoxLayout, QWidget, QApplication, QGroupBox, QHBoxLayout, QSpinBox, QMessageBox
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
        self.map_number = map_number or "Default"
        self.map_side = map_side or "Default"
        
        # Set audio folder
        self.audio_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hsm', 'audio')
        
        # Initialize variables for sentence tracking
        self.current_list = 1
        self.current_sentence = 1
        self.sentences_per_list = 20
        self.total_lists = 30
        self.snr = 10  # Default SNR value
        
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
        
        # Initialize UI
        self.initUI()
        
        # Initialize stream server status
        self.stream_server_running = True
    
    def center(self):
        """Center the window on the screen."""
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def initUI(self):
        """Initialize the user interface."""
        # Info label with map side
        infoLabel = QLabel(f"Subject ID: {self.id_val} | Map: {self.map_number} | Side: {self.map_side}")
        infoLabel.setAlignment(Qt.AlignCenter)
        infoLabel.setStyleSheet("color: white; font-size: 14px; padding: 10px; background-color: #2a2a2a; border-radius: 4px;")
        self.layout.addWidget(infoLabel)
        
        # Control Panel
        control_panel = QGroupBox("Test Control")
        control_panel.setStyleSheet("""
            QGroupBox {
                color: white;
                background-color: #34495e;
                border-radius: 5px;
                padding: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        control_layout = QVBoxLayout()
        
        # List selection
        list_layout = QHBoxLayout()
        list_label = QLabel("List:")
        self.list_combo = QComboBox()
        self.list_combo.addItems([f"List {i}" for i in range(1, self.total_lists + 1)])
        self.list_combo.currentIndexChanged.connect(self.onListChanged)
        list_layout.addWidget(list_label)
        list_layout.addWidget(self.list_combo)
        control_layout.addLayout(list_layout)
        
        # Sentence selection
        sentence_layout = QHBoxLayout()
        sentence_label = QLabel("Sentence:")
        self.sentence_combo = QComboBox()
        self.updateSentenceCombo()
        sentence_layout.addWidget(sentence_label)
        sentence_layout.addWidget(self.sentence_combo)
        control_layout.addLayout(sentence_layout)
        
        # SNR selection
        snr_layout = QHBoxLayout()
        snr_label = QLabel("SNR (dB):")
        self.snr_spin = QSpinBox()
        self.snr_spin.setRange(-20, 20)
        self.snr_spin.setValue(self.snr)
        self.snr_spin.valueChanged.connect(self.onSNRChanged)
        snr_layout.addWidget(snr_label)
        snr_layout.addWidget(self.snr_spin)
        control_layout.addLayout(snr_layout)
        
        # Play button
        self.play_button = QPushButton("Play Sentence")
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
        """)
        self.play_button.clicked.connect(self.playCurrentSentence)
        control_layout.addWidget(self.play_button)
        
        # Progress label
        self.progress_label = QLabel(f"Progress: List {self.current_list}/{self.total_lists}, Sentence {self.current_sentence}/{self.sentences_per_list}")
        self.progress_label.setStyleSheet("color: white;")
        control_layout.addWidget(self.progress_label)
        
        control_panel.setLayout(control_layout)
        self.layout.addWidget(control_panel)
        
        # Go Back button
        backButton = QPushButton("Go Back")
        backButton.clicked.connect(self.goBack)
        self.layout.addWidget(backButton)
        
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
    
    def onListChanged(self, index):
        """Handle list selection change."""
        self.current_list = index + 1
        self.current_sentence = 1
        self.updateSentenceCombo()
        self.updateProgressLabel()
        
    def onSNRChanged(self, value):
        """Handle SNR value change."""
        self.snr = value
        
    def updateSentenceCombo(self):
        """Update the sentence combo box with current list's sentences."""
        self.sentence_combo.clear()
        self.sentence_combo.addItems([f"Sentence {i}" for i in range(1, self.sentences_per_list + 1)])
        self.sentence_combo.setCurrentIndex(0)
        
    def updateProgressLabel(self):
        """Update the progress label with current position."""
        self.progress_label.setText(f"Progress: List {self.current_list}/{self.total_lists}, Sentence {self.current_sentence}/{self.sentences_per_list}")
        
    def playCurrentSentence(self):
        """Play the currently selected sentence."""
        # Get the current sentence number (1-based)
        sentence_num = self.sentence_combo.currentIndex() + 1
        
        # Calculate the file number:
        # For List X, Sentence Y, the file number is (X-1)*20+Y
        # Example: List 1, Sentence 1 should be (1-1)*20+1 = 1
        # Example: List 10, Sentence 5 should be (10-1)*20+5 = 185
        file_num = (self.current_list - 1) * self.sentences_per_list + sentence_num
        
        # Find the audio file for the current sentence - use exact matching to avoid confusion
        # between numbers like 1 and 10, or 10 and 100
        exact_pattern = f"HSM_{file_num}_"
        print(f"Looking for files with exact pattern: {exact_pattern}, from List {self.current_list}, Sentence {sentence_num}")
        
        # List all files in the audio folder for debugging
        print("All files in audio folder:")
        for f in os.listdir(self.audio_folder):
            print(f"  {f}")
            
        # Find files that match our pattern exactly and end with .wav
        audio_files = []
        for f in os.listdir(self.audio_folder):
            # Check if the file name is like HSM_X_... where X is exactly our file_num
            if f.endswith('.wav'):
                parts = f.split('_')
                if len(parts) >= 2 and parts[0] == "HSM" and parts[1] == str(file_num):
                    audio_files.append(f)
        
        print(f"Found {len(audio_files)} matching files:")
        for f in audio_files:
            print(f"  {f}")
        
        if not audio_files:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText(f"No audio file found for Sentence #{file_num} (List {self.current_list}, Sentence {sentence_num})")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #2c3e50;
                }
                QMessageBox QLabel {
                    color: #e74c3c;
                    font-size: 12px;
                    padding: 10px;
                }
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 5px 15px;
                    border-radius: 3px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            msg.exec_()
            return
            
        # Use the first matching file
        audio_file = audio_files[0]
        audio_path = os.path.join(self.audio_folder, audio_file)
        print(f"Using audio file: {audio_file}")
            
        try:
            # Stream the audio
            self.eng.stream(self.map_data, audio_path, nargout=0)
            
            # Update progress
            self.current_sentence = sentence_num + 1
            if self.current_sentence > self.sentences_per_list:
                self.current_sentence = 1
                self.current_list += 1
                if self.current_list > self.total_lists:
                    self.current_list = 1
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Complete")
                    msg.setText("All sentences have been played!")
                    msg.setStyleSheet("""
                        QMessageBox {
                            background-color: #2c3e50;
                        }
                        QMessageBox QLabel {
                            color: #2ecc71;
                            font-size: 12px;
                            padding: 10px;
                        }
                        QPushButton {
                            background-color: #2ecc71;
                            color: white;
                            border: none;
                            padding: 5px 15px;
                            border-radius: 3px;
                            min-width: 80px;
                        }
                        QPushButton:hover {
                            background-color: #27ae60;
                        }
                    """)
                    msg.exec_()
            
            # Update UI
            self.updateProgressLabel()
            self.sentence_combo.setCurrentIndex(self.current_sentence - 1)
            self.list_combo.setCurrentIndex(self.current_list - 1)
            
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to play audio: {str(e)}")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #2c3e50;
                }
                QMessageBox QLabel {
                    color: #e74c3c;
                    font-size: 12px;
                    padding: 10px;
                }
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 5px 15px;
                    border-radius: 3px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            msg.exec_()
    
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