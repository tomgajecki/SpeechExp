from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QApplication,
    QGroupBox, QHBoxLayout, QMessageBox, QGridLayout, QSlider, QProgressBar, QComboBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon
import os
import torchaudio
import matlab
from core.ui_components import CustomMessageBox

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

class HSMTestWindow(QMainWindow):
    """HSM test window."""
    
    # Signal to notify when window is closing
    window_closing = pyqtSignal()
    
    def __init__(self, eng, name, surname, id_val, map_data=None, map_number=None, map_side=None, ci_streaming_enabled=False):
        super().__init__()
        
        # Set window flags to prevent taskbar icon movement
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        
        # Store parameters
        self.eng = eng
        self.name = name
        self.surname = surname
        self.id_val = id_val
        self.map_data = map_data
        self.map_number = map_number or "Not Loaded"
        self.map_side = map_side or "N/A"

        self.ci_streaming_enabled = ci_streaming_enabled
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Flag to track if a map is loaded
        self.map_loaded = map_data is not None
        
        # Set up base folder
        self.base_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hsm')
        if not os.path.exists(self.base_folder):
            self.base_folder = os.path.join(os.getcwd(), "hsm")
        
        # Set up speech audio folder
        self.speech_folder = os.path.join(self.base_folder, 'audio', 'speech')
        if not os.path.exists(self.speech_folder):
            self.speech_folder = os.path.join(self.base_folder, "audio", "speech")
            
        # Set up noise folder
        self.noise_folder = os.path.join(self.base_folder, 'audio', 'noise')
        if not os.path.exists(self.noise_folder):
            self.noise_folder = os.path.join(self.base_folder, "audio", "noise")
        
        # Set up models folder
        self.models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        if not os.path.exists(self.models_folder):
            self.models_folder = os.path.join(os.getcwd(), "models")
        
        # Get available models
        self.available_models = ["Unprocessed"]  # Default option
        if os.path.exists(self.models_folder):
            for item in os.listdir(self.models_folder):
                if os.path.isdir(os.path.join(self.models_folder, item)):
                    self.available_models.append(item)
        
        # Current processing algorithm
        self.current_algorithm = "Unprocessed"
        
        # List configurations
        self.total_lists = 30
        self.sentences_per_list = 20
        self.current_list = 1
        self.current_sentence = 1
        self.noise_type = "icra7"  # Default noise type
        
        # Add MATLAB functions directory to path
        matlab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'matlab')
        self.eng.addpath(matlab_path, nargout=0)
        
        # Add flag for audio playback state
        self.is_playing = False
        
        self.initUI()
        self.stream_server_running = True
        
        # Show warning if no map is loaded
        if not self.map_loaded:
            QTimer.singleShot(500, self.showMapWarning)  # Show warning after window is fully loaded
        
    def initUI(self):
        self.setWindowTitle("HSM Test")
        # Use a smaller size
        self.resize(950, 650)
        self.setMaximumHeight(650)
        
        # Main central widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setSpacing(8)  # Further reduce spacing
        self.layout.setContentsMargins(10, 10, 10, 10)  # Smaller margins
        
        # Subject info header
        subject_info = QLabel(f"Subject ID: {self.id_val}   Map: {self.map_number}   Side: {self.map_side}")
        subject_info.setStyleSheet("""
            background-color: #e3f2fd;
            color: #000000;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin: 2px;
        """)
        subject_info.setMinimumHeight(32)
        self.layout.addWidget(subject_info)
        

        # Control panel container
        control_container = QWidget()
        control_container.setStyleSheet("""
            QWidget {
                background-color: #1e2b38;
                border-radius: 5px;
                padding: 6px;
            }
            QLabel { color: white; font-size: 12px; }
            QGroupBox {
                color: white;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #3498db;
                border-radius: 3px;
                margin-top: 10px;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 6px;
            }
        """)
        control_layout = QVBoxLayout(control_container)
        control_layout.setSpacing(6)
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        self.controlPanel = self.setupControlPanel()
        control_layout.addWidget(self.controlPanel)
        self.layout.addWidget(control_container)
        
        # Go Back button container
        back_container = QWidget()
        back_layout = QHBoxLayout(back_container)
        back_layout.setContentsMargins(0, 5, 0, 0)
        self.back_btn = QPushButton("Go Back")
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self.back_btn.setFixedSize(80, 26)
        self.back_btn.clicked.connect(self.goBack)
        back_layout.addStretch()
        back_layout.addWidget(self.back_btn)
        self.layout.addWidget(back_container)

        self.main_widget.setStyleSheet("background-color: #2c3e50; color: white;")
    
    def setupControlPanel(self):
        panel = QWidget()
        control_layout = QVBoxLayout(panel)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(10, 8, 10, 8)
        
        # Algorithm Selection
        algo_group = QGroupBox("Processing Algorithm")
        algo_group.setStyleSheet("""
            QGroupBox { 
                font-size: 12px; 
                font-weight: bold; 
                padding-top: 10px; 
                margin-top: 6px; 
            }
        """)
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setContentsMargins(6, 10, 6, 6)
        algo_layout.setSpacing(6)
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(self.available_models)
        self.algo_combo.setCurrentText("Unprocessed")
        self.algo_combo.setStyleSheet("""
            QComboBox {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #34495e;
                border-radius: 3px;
                padding: 5px;
                min-height: 24px;
                font-size: 12px;
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
            QComboBox:hover {
                border: 1px solid #3498db;
            }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
                selection-color: white;
                border: 1px solid #34495e;
            }
        """)
        self.algo_combo.currentTextChanged.connect(self.onAlgorithmChanged)
        
        algo_layout.addWidget(self.algo_combo)
        control_layout.addWidget(algo_group)
        
        # Selection Panel group
        selection_group = QGroupBox("Selection Panel")
        selection_group.setStyleSheet("""
            QGroupBox { font-size: 13px; font-weight: bold; padding-top: 14px; }
        """)
        selection_layout = QVBoxLayout(selection_group)
        selection_layout.setSpacing(2)  
        selection_layout.setContentsMargins(6, 6, 6, 6)
        
        # LISTS header
        list_label = QLabel("LISTS")
        list_label.setStyleSheet("""
            font-weight: bold; color: #3498db; font-size: 12px;
            padding: 3px; background-color: rgba(52, 152, 219, 0.2);
            border: 1px solid #3498db;
        """)
        list_label.setMinimumHeight(22)
        selection_layout.addWidget(list_label)
        
        # Grid layout for list buttons
        lists_grid = QGridLayout()
        lists_grid.setHorizontalSpacing(2)
        lists_grid.setVerticalSpacing(2)
        lists_grid.setContentsMargins(0, 2, 0, 2)
        
        self.list_buttons = []
        # Use 6 columns so that 30 buttons fit in 5 rows
        columns = 6
        for i in range(1, self.total_lists + 1):
            btn = QPushButton(f"{i}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #213141;
                    color: white;
                    border: 1px solid #95a5a6;
                    font-size: 8px;
                    font-weight: bold;
                    padding: 1px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                    border: 1px solid #3498db;
                }
                QPushButton:checked {
                    background-color: #3498db;
                    border: 1px solid white;
                    font-weight: bold;
                    color: white;
                }
            """)
            btn.setFixedSize(30, 22)
            btn.setCheckable(True)
            if i == 1:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, list_num=i: self.setCurrentList(list_num))
            lists_grid.addWidget(btn, (i-1) // columns, (i-1) % columns)
            self.list_buttons.append(btn)
        selection_layout.addLayout(lists_grid)
        
        # SENTENCES header
        sentence_label = QLabel("SENTENCES")
        sentence_label.setStyleSheet("""
            font-weight: bold; color: #2ecc71; font-size: 12px;
            padding: 3px; background-color: rgba(46, 204, 113, 0.2);
            border: 1px solid #2ecc71;
            margin-top: 4px;
        """)
        sentence_label.setMinimumHeight(22)
        selection_layout.addWidget(sentence_label)
        
        # Grid layout for sentence buttons
        sentences_grid = QGridLayout()
        sentences_grid.setHorizontalSpacing(2)
        sentences_grid.setVerticalSpacing(2)
        sentences_grid.setContentsMargins(0, 2, 0, 2)
        
        self.sentence_buttons = []
        # Use 5 columns so that 20 buttons fit in 4 rows
        columns = 5
        for i in range(1, self.sentences_per_list + 1):
            btn = QPushButton(f"{i}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #213141;
                    color: white;
                    border: 1px solid #95a5a6;
                    font-size: 8px;
                    font-weight: bold;
                    padding: 1px;
                }
                QPushButton:hover {
                    background-color: #27ae60;
                    border: 1px solid #2ecc71;
                }
                QPushButton:checked {
                    background-color: #2ecc71;
                    border: 1px solid white;
                    font-weight: bold;
                    color: white;
                }
            """)
            btn.setFixedSize(30, 22)
            btn.setCheckable(True)
            if i == 1:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, sentence_num=i: self.setCurrentSentence(sentence_num))
            sentences_grid.addWidget(btn, (i-1) // columns, (i-1) % columns)
            self.sentence_buttons.append(btn)
        selection_layout.addLayout(sentences_grid)
        
        # SNR Control
        snr_group = QGroupBox("SNR Control")
        snr_group.setStyleSheet("""
            QGroupBox { font-size: 12px; font-weight: bold; padding-top: 10px; margin-top: 6px; }
        """)
        snr_layout = QVBoxLayout(snr_group)
        snr_layout.setContentsMargins(6, 10, 6, 6)
        snr_layout.setSpacing(4)
        
        # SNR Slider layout
        snr_slider_layout = QHBoxLayout()
        
        snr_label = QLabel("SNR (dB):")
        snr_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        snr_label.setMinimumWidth(50)
        
        # Create minus button for SNR
        self.snr_minus_btn = QPushButton("-")
        self.snr_minus_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                border-radius: 2px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #2c3e50; }
            QPushButton:pressed { background-color: #1a2530; }
        """)
        self.snr_minus_btn.setFixedSize(24, 22)
        self.snr_minus_btn.clicked.connect(self.decrementSNR)
        
        slider_container = QWidget()
        slider_container.setMinimumWidth(180)
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        
        self.snr_slider = QSlider(Qt.Horizontal)
        self.snr_slider.setRange(-10, 20)
        self.snr_slider.setValue(0)
        self.snr_slider.setTickPosition(QSlider.TicksBelow)
        self.snr_slider.setTickInterval(5)
        self.snr_slider.setFixedHeight(22)
        
        # Create plus button for SNR
        self.snr_plus_btn = QPushButton("+")
        self.snr_plus_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                border-radius: 2px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #2c3e50; }
            QPushButton:pressed { background-color: #1a2530; }
        """)
        self.snr_plus_btn.setFixedSize(24, 22)
        self.snr_plus_btn.clicked.connect(self.incrementSNR)
        
        self.snr_value = QLabel("0 dB")
        self.snr_value.setStyleSheet("font-size: 12px; font-weight: bold; min-width: 40px;")
        self.snr_value.setAlignment(Qt.AlignCenter)
        self.snr_slider.valueChanged.connect(self.updateSNRValue)
        
        # Noise selection layout
        noise_layout = QHBoxLayout()
        
        noise_label = QLabel("Noise Type:")
        noise_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        noise_label.setMinimumWidth(60)
        
        self.icra7_btn = QPushButton("ICRA7")
        self.icra7_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 3px 8px;
                border-radius: 2px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #c0392b; }
            QPushButton:checked {
                background-color: #c0392b;
                border: 1px solid white;
            }
        """)
        self.icra7_btn.setFixedHeight(24)
        
        self.ccitt_btn = QPushButton("CCITT")
        self.ccitt_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 3px 8px;
                border-radius: 2px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #c0392b; }
            QPushButton:checked {
                background-color: #c0392b;
                border: 1px solid white;
            }
        """)
        self.ccitt_btn.setFixedHeight(24)
        
        self.clean_btn = QPushButton("Clean")
        self.clean_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 3px 8px;
                border-radius: 2px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #27ae60; }
            QPushButton:checked {
                background-color: #27ae60;
                border: 1px solid white;
            }
        """)
        self.clean_btn.setFixedHeight(24)
        self.clean_btn.setCheckable(True)
        
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(self.icra7_btn)
        noise_layout.addWidget(self.ccitt_btn)
        noise_layout.addWidget(self.clean_btn)
        noise_layout.addStretch(1)
        
        # Add both layouts to SNR control group
        snr_layout.addLayout(snr_slider_layout)
        snr_layout.addLayout(noise_layout)
        
        # Initialize current values
        self.current_list = 1
        self.current_sentence = 1
        
        # Set up slider
        slider_layout.addWidget(self.snr_slider)
        
        snr_slider_layout.addWidget(snr_label)
        snr_slider_layout.addWidget(self.snr_minus_btn)
        snr_slider_layout.addWidget(slider_container)
        snr_slider_layout.addWidget(self.snr_plus_btn)
        snr_slider_layout.addWidget(self.snr_value)
        snr_slider_layout.addStretch(1)
        
        # Set up noise buttons
        self.icra7_btn.setCheckable(True)
        self.icra7_btn.setChecked(True)
        self.icra7_btn.clicked.connect(lambda: self.setNoiseType("icra7"))
        
        self.ccitt_btn.setCheckable(True)
        self.ccitt_btn.clicked.connect(lambda: self.setNoiseType("ccitt"))
        
        self.clean_btn.setCheckable(True)
        self.clean_btn.clicked.connect(lambda: self.setNoiseType("clean"))
        
        # Add all groups to the control panel layout
        control_layout.addWidget(selection_group, 5)
        control_layout.addWidget(snr_group, 2)
        
        # Playback Control
        play_group = QGroupBox("Playback Control")
        play_group.setStyleSheet("""
            QGroupBox { font-size: 12px; font-weight: bold; padding-top: 10px; margin-top: 6px; }
        """)
        play_layout = QVBoxLayout(play_group)
        play_layout.setContentsMargins(6, 10, 6, 6)
        play_layout.setSpacing(6)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.play_btn = QPushButton("Play")
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #27ae60; }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #7f8c8d;
            }
        """)
        self.play_btn.setFixedSize(120, 30)  # Set fixed width and height
        self.play_btn.setEnabled(self.map_loaded)  # Disable if no map is loaded
        
        # Playing indicator
        self.playing_label = QLabel("")
        self.playing_label.setStyleSheet("""
            color: #2ecc71;
            font-weight: bold;
            font-size: 12px;
            padding: 4px;
            background-color: rgba(46, 204, 113, 0.1);
            border-radius: 3px;
        """)
        self.playing_label.setAlignment(Qt.AlignCenter)
        self.playing_label.setMinimumHeight(24)
        
        # Add stretch before and after the button to center it
        button_layout.addStretch(1)
        button_layout.addWidget(self.play_btn)
        button_layout.addStretch(1)
        
        play_layout.addLayout(button_layout)
        play_layout.addWidget(self.playing_label)
        
        # Connect button signal
        self.play_btn.clicked.connect(self.playAndAdvance)
        
        control_layout.addWidget(play_group, 2)
        control_layout.addStretch(1)
        
        return panel
        
    def updateProgressLabel(self):
        """Update the progress label with current list and sentence information."""
        # Style the entire List part blue
        list_part = f'<span style="color: #2196F3; font-weight: bold; font-size: 12px;">List {self.current_list} / {self.total_lists}</span>'
        # Style the entire Sentence part green
        sentence_part = f'<span style="color: #4CAF50; font-weight: bold; font-size: 12px;">Sentence {self.current_sentence} / {self.sentences_per_list}</span>'

        # Combine the styled parts with a comma separator - removed "Playing..." logic
        self.playing_label.setText(f"{list_part}, {sentence_part}")

    def setCurrentList(self, list_num):
        if 1 <= list_num <= self.total_lists:
            self.current_list = list_num
            for i, btn in enumerate(self.list_buttons):
                btn.setChecked(i + 1 == list_num)
            # Reset sentence number to 1 when changing lists
            self.setCurrentSentence(1)
            self.updateProgressLabel()
    
    def setCurrentSentence(self, sentence_num):
        if 1 <= sentence_num <= self.sentences_per_list:
            self.current_sentence = sentence_num
            for i, btn in enumerate(self.sentence_buttons):
                btn.setChecked(i + 1 == sentence_num)
            self.updateProgressLabel()
            
    def nextSentence(self):
        next_sentence = self.current_sentence + 1
        if next_sentence > self.sentences_per_list:
            next_sentence = 1
            next_list = self.current_list + 1
            if next_list > self.total_lists:
                next_list = 1
            self.setCurrentList(next_list)
        self.setCurrentSentence(next_sentence)
        
    def updateSNRValue(self, value):
        self.snr_value.setText(f"{value} dB")
        
    def disablePlaybackControls(self):
        """Disable all playback-related controls."""
        self.play_btn.setEnabled(False)
        self.icra7_btn.setEnabled(False)
        self.ccitt_btn.setEnabled(False)
        self.clean_btn.setEnabled(False)
        self.snr_slider.setEnabled(False)
        self.snr_minus_btn.setEnabled(False)
        self.snr_plus_btn.setEnabled(False)

    def enablePlaybackControls(self):
        """Enable all playback-related controls."""
        self.play_btn.setEnabled(self.map_loaded)
        self.icra7_btn.setEnabled(True)
        self.ccitt_btn.setEnabled(True)
        self.clean_btn.setEnabled(True)
        self.snr_slider.setEnabled(True)
        self.snr_minus_btn.setEnabled(True)
        self.snr_plus_btn.setEnabled(True)

    def playCurrentSentence(self, with_noise=True):
        """Initiates playback: updates GUI, then schedules streaming via timer."""
        # Prevent multiple playbacks
        if self.is_playing:
            print("Playback already in progress.")
            return False # Indicate busy state

        # Check if a map is loaded
        if not self.map_loaded:
            msg = CustomMessageBox(
                self,
                "No Map Loaded",
                "No map is currently loaded.",
                QMessageBox.Warning,
                "Please load a map in the main window before playing audio."
            )
            msg.exec_()
            return False # Indicate failure

        # --- Start Playback Sequence ---
        self.is_playing = True
        self.disablePlaybackControls()

        # --- GUI Update First ---
        print("Updating GUI for current sentence before scheduling stream...")
        self.updateProgressLabel() # Show the current sentence being prepared
        QApplication.processEvents() # Ensure GUI updates visually

        # --- Schedule Streaming ---
        # Use QTimer.singleShot to decouple streaming from the initial GUI update
        # Pass necessary parameters (like with_noise) via lambda if needed
        print("Scheduling streaming logic...")
        QTimer.singleShot(0, lambda: self._execute_streaming(with_noise))

        # Return immediately, _execute_streaming will handle the rest
        return True # Indicate initiation success

    def _execute_streaming(self, with_noise):
        """Handles the actual audio loading, processing, and streaming logic."""
        print(f"Executing streaming for List {self.current_list}, Sentence {self.current_sentence}")
        speech_path = "N/A"
        noise_path = "N/A"
        snr = "N/A"
        playback_successful = False
        status_msg = "Playback status unknown"
        stream_log_msg = "Stream log message not set"
        audio_to_stream = None # Initialize

        try:
            # Get current sentence info (needed again here as it runs later)
            sentence_num = self.current_sentence
            list_num = self.current_list
            file_num = (list_num - 1) * self.sentences_per_list + sentence_num

            # --- Audio Loading --- (Moved outside conditional logic for clarity)
            # Check if speech folder exists
            if not os.path.exists(self.speech_folder):
                msg = CustomMessageBox(self, "Error", f"Speech folder not found: {self.speech_folder}", QMessageBox.Warning)
                msg.exec_()
                raise FileNotFoundError(f"Speech folder not found: {self.speech_folder}")

            # Find speech audio file
            audio_files = []
            for f in os.listdir(self.speech_folder):
                if f.endswith('.wav'):
                    parts = f.split('_')
                    if len(parts) >= 2 and parts[0] == "HSM" and parts[1] == str(file_num):
                        audio_files.append(f)
            if not audio_files:
                 msg = CustomMessageBox(self, "Error", f"No audio file found for Sentence #{file_num} (List {list_num}, Sentence {sentence_num})", QMessageBox.Warning, f"Searched in: {self.speech_folder}")
                 msg.exec_()
                 raise FileNotFoundError(f"Audio file not found for sentence {file_num}")

            speech_file = audio_files[0]
            speech_path = os.path.join(self.speech_folder, speech_file)
            speech_tensor, fs_speech = torchaudio.load(speech_path)
            self.fs_speech = fs_speech
            speech_data = speech_tensor.numpy()
            if len(speech_data.shape) > 1 and speech_data.shape[0] > 1:
                speech_data = speech_data[0, :]
            speech_data_matlab = matlab.double(speech_data.tolist())

            # --- Prepare Audio To Stream based on conditions ---
            if with_noise:
                # Check if noise folder exists
                if not os.path.exists(self.noise_folder):
                    msg = CustomMessageBox(self, "Error", f"Noise folder not found: {self.noise_folder}", QMessageBox.Warning)
                    msg.exec_()
                    raise FileNotFoundError(f"Noise folder not found: {self.noise_folder}")

                noise_file = f"{self.noise_type}.wav"
                noise_path = os.path.join(self.noise_folder, noise_file)
                if not os.path.exists(noise_path):
                     msg = CustomMessageBox(self, "Error", f"Noise file not found: {noise_file}", QMessageBox.Warning, f"Searched in: {self.noise_folder}")
                     msg.exec_()
                     raise FileNotFoundError(f"Noise file not found: {noise_file}")

                noise_tensor, fs_noise = torchaudio.load(noise_path)
                if fs_speech != fs_noise:
                     raise ValueError(f"Sampling rates mismatch: Speech {fs_speech}Hz, Noise {fs_noise}Hz")
                noise_data = noise_tensor.numpy()
                if len(noise_data.shape) > 1 and noise_data.shape[0] > 1:
                     noise_data = noise_data[0, :]
                noise_data_matlab = matlab.double(noise_data.tolist())
                snr = self.snr_slider.value()
                mixed_audio = self.eng.mix_audio(speech_data_matlab, noise_data_matlab, float(snr), float(fs_speech), nargout=1)
                audio_to_stream = mixed_audio

                if self.current_algorithm != "Unprocessed":
                    print(f"Processing mixed audio with {self.current_algorithm}...")
                    processed_signal = self.processAudio(mixed_audio, self.current_algorithm)
                    audio_to_stream = processed_signal
                    status_msg = f"Played processed sentence {sentence_num} list {list_num} ({self.noise_type.upper()}@{snr}dB) [{self.current_algorithm}]"
                    stream_log_msg = f"Streaming processed mix: L{list_num} S{sentence_num}, Algo: {self.current_algorithm}, Noise: {self.noise_type.upper()}, SNR: {snr} dB"
                else:
                    status_msg = f"Played sentence {sentence_num} list {list_num} ({self.noise_type.upper()}@{snr}dB) [Unprocessed]"
                    stream_log_msg = f"Streaming unprocessed mix: L{list_num} S{sentence_num}, Noise: {self.noise_type.upper()}, SNR: {snr} dB"
            else: # Clean
                audio_to_stream = speech_data_matlab
                if self.current_algorithm != "Unprocessed":
                    print(f"Processing clean audio with {self.current_algorithm}...")
                    processed_audio = self.processAudio(speech_data_matlab, self.current_algorithm)
                    audio_to_stream = processed_audio
                    status_msg = f"Played processed clean sentence {sentence_num} list {list_num} [{self.current_algorithm}]"
                    stream_log_msg = f"Streaming processed clean: L{list_num} S{sentence_num}, Algo: {self.current_algorithm}"
                else:
                    status_msg = f"Played clean sentence {sentence_num} list {list_num} [Unprocessed]"
                    stream_log_msg = f"Streaming clean: L{list_num} S{sentence_num}"

            # Ensure audio_to_stream is assigned
            if audio_to_stream is None:
                 raise ValueError("audio_to_stream was not assigned. Check logic.")

            # --- Streaming --- 
            self.setStatusText(status_msg)
            print(f"Calling eng.stream: {stream_log_msg}")
            self.eng.stream(self.map_data, audio_to_stream, self.ci_streaming_enabled, nargout=0)
            print("eng.stream call returned.")
            

            playback_successful = True # Mark as successful if streaming call completed without error

        except Exception as e:
            import traceback
            print("--- ERROR DURING STREAMING --- ")
            traceback.print_exc() 
            print("--- END ERROR --- ")
            msg = CustomMessageBox(
                self, "Error During Playback", f"Failed to play audio: {str(e)}",
                QMessageBox.Critical,
                f"Details:\nSpeech file: {speech_path}\nNoise file: {noise_path}\nSNR: {snr} dB"
            )
            msg.exec_()
            # Error state handled by scheduling _finalize_playback with success=False

        finally:
            # --- Schedule Finalization --- 
            # This always runs, whether try block succeeded or failed.
            # Schedule the final GUI updates and state reset slightly later.
            delay_ms = 200 # Add a small delay (e.g., 200ms)
            print(f"Scheduling finalization step with success={playback_successful} after {delay_ms}ms delay...")
            QTimer.singleShot(delay_ms, lambda: self._finalize_playback(playback_successful))
            # Note: is_playing remains True until _finalize_playback runs

    def _finalize_playback(self, success):
        """Final step after streaming attempt: Re-enable controls, advance sentence if successful."""
        print(f"Executing finalization step. Success: {success}")
        
        if not self.is_playing:
             print("Finalization warning: is_playing was already false.")
             # Might happen if user interacts rapidly or error occurs unexpectedly
             # Still attempt to enable controls just in case.
             self.enablePlaybackControls()
             return # Avoid double updates

        self.is_playing = False
        self.enablePlaybackControls()
        
        if success:
            print("Playback successful, advancing sentence and updating GUI...")
            self.nextSentence() # This calls updateProgressLabel for the *next* sentence
        else:
            print("Playback failed or error occurred, updating GUI to reflect current selection...")
            self.updateProgressLabel() # Ensure label reflects the current sentence after failed attempt
        
        QApplication.processEvents() # Force GUI update after advancing or on failure recovery
        print("Finalization complete.")

    def processAudio(self, audio_data, algorithm):
        """Process audio data using the selected algorithm."""
        if algorithm == "Unprocessed":
            return audio_data
            
        if algorithm == "Wiener":
            try:
                # Process using MATLAB Wiener function
                return self.eng.wiener_process(audio_data[0], self.fs_speech, nargout=1)
            except Exception as e:
                print(f"Error in Wiener processing: {e}")
                return audio_data
            
        # Get the model path and config path for the selected algorithm
        model_dir = os.path.join(self.models_folder, algorithm)
        
        # Check if the model directory exists
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            return audio_data
            
        try:
            # Import the run_inference function from single_inference.py
            from single_inference import run_inference
            
            # Convert audio_data to numpy array if it's a MATLAB array
            if hasattr(audio_data, 'dtype') and hasattr(audio_data, 'shape'):
                # It's already a numpy array
                numpy_audio = audio_data
            else:
                # Convert MATLAB array to numpy array
                import numpy as np
                numpy_audio = np.array(audio_data)
            
            # Force garbage collection to clear any previous models
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
            
            print(f"Processing audio with algorithm: {algorithm}")
            print(f"Model directory: {model_dir}")
            
            # Check if the model directory contains the necessary files
            if not os.path.exists(os.path.join(model_dir, "checkpoint")):
                print(f"Checkpoint directory not found in {model_dir}")
                return audio_data
                
            # Check for .pth files in the checkpoint directory
            checkpoint_dir = os.path.join(model_dir, "checkpoint")
            pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if not pth_files:
                print(f"No .pth files found in {checkpoint_dir}")
                return audio_data
                
            print(f"Found model files: {pth_files}")
            
            # Check for modules directory
            modules_dir = os.path.join(model_dir, "modules")
            if not os.path.exists(modules_dir):
                print(f"Modules directory not found in {model_dir}")
                return audio_data
                
            # Check for model.py in modules directory
            if not os.path.exists(os.path.join(modules_dir, "model.py")):
                print(f"model.py not found in {modules_dir}")
                return audio_data
            
            # Run inference using the model directory
            processed_audio = run_inference(
                waveform=numpy_audio,
                model_dir=model_dir,
                model_type=algorithm
            )
            
            # Convert back to MATLAB array if needed
            import matlab
            processed_audio_matlab = matlab.double(processed_audio.tolist())
            
            return processed_audio_matlab
            
        except Exception as e:
            print(f"Error processing audio with {algorithm}: {e}")
            import traceback
            traceback.print_exc()
            return audio_data

    def goBack(self):
        self.hide()
        for widget in QApplication.topLevelWidgets():
            if widget.objectName() == "mainWindow":
                widget.show()
                break
    
    def closeEvent(self, event):
        self.window_closing.emit()
        event.accept() 

    def setStatusText(self, text):
        self.statusBar().showMessage(text)

    def setNoiseType(self, noise_type):
        """Update the noise type and button states."""
        self.noise_type = noise_type
        
        # Update button checked states
        self.icra7_btn.setChecked(noise_type == "icra7")
        self.ccitt_btn.setChecked(noise_type == "ccitt")
        self.clean_btn.setChecked(noise_type == "clean")
        
        # Enable/disable SNR controls based on noise type
        snr_enabled = noise_type != "clean"
        self.snr_slider.setEnabled(snr_enabled)
        self.snr_minus_btn.setEnabled(snr_enabled)
        self.snr_plus_btn.setEnabled(snr_enabled)
        self.snr_value.setEnabled(snr_enabled)
        
        # Update status bar
        self.setStatusText(f"Noise type set to {noise_type.upper()}")

    def decrementSNR(self):
        current_value = self.snr_slider.value()
        if current_value > -10:
            self.snr_slider.setValue(current_value - 5)

    def incrementSNR(self):
        current_value = self.snr_slider.value()
        if current_value < 20:
            self.snr_slider.setValue(current_value + 5)

    def playAndAdvance(self):
        """Plays the current sentence, advancement is handled internally."""
        print("Play button clicked. Initiating playback sequence...")
        self.playCurrentSentence(with_noise=(self.noise_type != "clean"))

    def loadNextSentence(self):
        """Load the next sentence from the list."""
        try:
            if not self.sentence_files:
                self.showWarning("Warning", "No more sentences available.")
                return False

            self.current_sentence = self.sentence_files.pop(0)
            return True
        except Exception as e:
            self.showError("Error", f"Error loading next sentence: {e}")
            return False

    def saveResults(self):
        """Save test results."""
        try:
            # Save results implementation
            self.showInfo("Success", "Results saved successfully!")
        except Exception as e:
            self.showError("Error", f"Error saving results: {e}")

    def onAlgorithmChanged(self, algorithm):
        """Handle algorithm selection change."""
        self.current_algorithm = algorithm
        print(f"Selected algorithm: {algorithm}")  # For debugging
        # We'll add the processing logic later

    def showMapWarning(self):
        """Show a warning message about loading a map."""
        msg = CustomMessageBox(
            self,
            "No Map Loaded",
            "No map is currently loaded.",
            QMessageBox.Warning,
            "Please load a map in the main window before starting the experiment."
        )
        msg.exec_()

    def showError(self, title, message):
        """Show error message box."""
        msg = CustomMessageBox(
            self,
            title,
            message,
            QMessageBox.Critical
        )
        msg.exec_()

    def showWarning(self, title, message):
        """Show warning message box."""
        msg = CustomMessageBox(
            self,
            title,
            message,
            QMessageBox.Warning
        )
        msg.exec_()

    def showInfo(self, title, message):
        """Show information message box."""
        msg = CustomMessageBox(
            self,
            title,
            message,
            QMessageBox.Information
        )
        msg.exec_()
