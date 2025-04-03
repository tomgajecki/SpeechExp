import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLineEdit, QLabel, 
                            QPushButton, QComboBox, QGridLayout, QVBoxLayout, QSpacerItem, 
                            QSizePolicy, QMessageBox, QSplashScreen, QFileDialog, QProgressBar)
from PyQt5.QtGui import QFont, QPixmap, QMovie, QColor
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QRect, QEasingCurve

from core.matlab_engine import MatlabEngineLoader
from core.base_window import BaseWindow
from hsm.hsm_window import HSMTestWindow
from olsa.olsa_window import OLSATestWindow

class MainWindow(BaseWindow):
    """Main window for the Speech Intelligibility Experiment."""
    
    def __init__(self, engine):
        super().__init__(None, "Speech Intelligibility Experiment")
        self.eng = engine
        self.map_data = None
        self.initUI()
        # Don't show the window immediately
        self.hide()
        self.setObjectName("mainWindow")  # Set object name for finding the window
        self.setFixedSize(800, 600)
        self.center()
        
        # Apply styles
        self.applyStyles()
    
    def initUI(self):
        super().initUI()
        
        # Title
        titleLabel = QLabel("Speech Intelligibility Experiment")
        titleFont = QFont("Arial", 22, QFont.Bold)
        titleLabel.setFont(titleFont)
        titleLabel.setAlignment(Qt.AlignCenter)
        titleLabel.setStyleSheet("color: white;")
        self.layout.addWidget(titleLabel)
        
        # Subject Information Section
        subjectGroupLabel = QLabel("Subject Information")
        subjectGroupLabel.setFont(QFont("Arial", 16, QFont.Bold))
        subjectGroupLabel.setStyleSheet("color: white;")
        self.layout.addWidget(subjectGroupLabel)
        
        subjectGrid = QGridLayout()
        subjectGrid.setSpacing(10)
        
        nameLabel = QLabel("Name:")
        nameLabel.setStyleSheet("color: white;")
        self.nameInput = QLineEdit("John")
        
        surnameLabel = QLabel("Surname:")
        surnameLabel.setStyleSheet("color: white;")
        self.surnameInput = QLineEdit("Doe")
        
        idLabel = QLabel("ID:")
        idLabel.setStyleSheet("color: white;")
        self.idInput = QLineEdit("12345")
        
        testLabel = QLabel("Select Test:")
        testLabel.setStyleSheet("color: white;")
        self.testCombo = QComboBox()
        self.testCombo.addItems(["HSM", "OLSA"])
        
        subjectGrid.addWidget(nameLabel, 0, 0)
        subjectGrid.addWidget(self.nameInput, 0, 1)
        subjectGrid.addWidget(surnameLabel, 1, 0)
        subjectGrid.addWidget(self.surnameInput, 1, 1)
        subjectGrid.addWidget(idLabel, 2, 0)
        subjectGrid.addWidget(self.idInput, 2, 1)
        subjectGrid.addWidget(testLabel, 3, 0)
        subjectGrid.addWidget(self.testCombo, 3, 1)
        
        self.layout.addLayout(subjectGrid)
        
        # Separator
        separator1 = QLabel()
        separator1.setFixedHeight(1)
        separator1.setStyleSheet("background-color: #5e5e5e;")
        self.layout.addWidget(separator1)
        
        # Map Settings Section
        mapGroupLabel = QLabel("CI Map Settings")
        mapGroupLabel.setFont(QFont("Arial", 16, QFont.Bold))
        mapGroupLabel.setStyleSheet("color: white;")
        self.layout.addWidget(mapGroupLabel)
        
        mapGrid = QGridLayout()
        mapGrid.setSpacing(10)
        
        mapSideLabel = QLabel("Hearing Side:")
        mapSideLabel.setStyleSheet("color: white;")
        self.sideCombo = QComboBox()
        self.sideCombo.addItems(["Left", "Right"])
        
        mapNumberLabel = QLabel("Map Number:")
        mapNumberLabel.setStyleSheet("color: white;")
        self.mapNumberEdit = QLineEdit("1")
        
        mapGrid.addWidget(mapSideLabel, 0, 0)
        mapGrid.addWidget(self.sideCombo, 0, 1)
        mapGrid.addWidget(mapNumberLabel, 1, 0)
        mapGrid.addWidget(self.mapNumberEdit, 1, 1)
        
        self.layout.addLayout(mapGrid)
        
        self.loadMapButton = QPushButton("Load Map")
        self.loadMapButton.clicked.connect(self.loadMap)
        self.layout.addWidget(self.loadMapButton)
        
        # Separator
        separator2 = QLabel()
        separator2.setFixedHeight(1)
        separator2.setStyleSheet("background-color: #5e5e5e;")
        self.layout.addWidget(separator2)
        
        # Map Data Section
        self.mapDataGroupLabel = QLabel("CI Map Data")
        self.mapDataGroupLabel.setFont(QFont("Arial", 16, QFont.Bold))
        self.mapDataGroupLabel.setStyleSheet("color: white;")
        self.layout.addWidget(self.mapDataGroupLabel)
        
        self.mapDataLayout = QGridLayout()
        self.mapDataLayout.setVerticalSpacing(25)
        self.mapDataLayout.setHorizontalSpacing(10)
        
        thresholdLabel = QLabel("Thresholds (T Levels):")
        thresholdLabel.setStyleSheet("color: white;")
        self.lowerLevelsLabel = QLabel("Not loaded")
        self.lowerLevelsLabel.setWordWrap(True)
        self.lowerLevelsLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lowerLevelsLabel.setStyleSheet("background-color: #2a2a2a; padding: 8px; border-radius: 3px; color: white;")
        
        comfortableLabel = QLabel("Comfortable Levels (C Levels):")
        comfortableLabel.setStyleSheet("color: white;")
        self.upperLevelsLabel = QLabel("Not loaded")
        self.upperLevelsLabel.setWordWrap(True)
        self.upperLevelsLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.upperLevelsLabel.setStyleSheet("background-color: #2a2a2a; padding: 8px; border-radius: 3px; color: white;")
        
        stimRateLabel = QLabel("Channel Stimulation Rate (Hz):")
        stimRateLabel.setStyleSheet("color: white;")
        self.channelStimRateLabel = QLabel("Not loaded")
        self.channelStimRateLabel.setWordWrap(True)
        self.channelStimRateLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.channelStimRateLabel.setStyleSheet("background-color: #2a2a2a; padding: 8px; border-radius: 3px; color: white;")
        
        self.mapDataLayout.addWidget(thresholdLabel, 0, 0, Qt.AlignTop)
        self.mapDataLayout.addWidget(self.lowerLevelsLabel, 0, 1)
        self.mapDataLayout.addWidget(comfortableLabel, 1, 0, Qt.AlignTop)
        self.mapDataLayout.addWidget(self.upperLevelsLabel, 1, 1)
        self.mapDataLayout.addWidget(stimRateLabel, 2, 0, Qt.AlignTop)
        self.mapDataLayout.addWidget(self.channelStimRateLabel, 2, 1)
        
        self.mapDataLayout.setColumnStretch(1, 1)
        self.layout.addLayout(self.mapDataLayout)
        
        # Hide map data section initially
        self.mapDataGroupLabel.setVisible(False)
        for i in range(self.mapDataLayout.count()):
            item = self.mapDataLayout.itemAt(i)
            if item.widget():
                item.widget().setVisible(False)
        
        # Spacer
        self.layout.addSpacing(20)
        self.layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Action Buttons
        btnLayout = QGridLayout()
        btnLayout.setSpacing(20)
        
        self.startButton = QPushButton("Start Experiment")
        self.helpButton = QPushButton("Help")
        self.exitButton = QPushButton("Exit")
        
        btnLayout.addWidget(self.startButton, 0, 0)
        btnLayout.addWidget(self.helpButton, 0, 1)
        btnLayout.addWidget(self.exitButton, 0, 2)
        
        self.layout.addLayout(btnLayout)
        
        # Connect signals
        self.startButton.clicked.connect(self.startTest)
        self.helpButton.clicked.connect(self.showHelp)
        self.exitButton.clicked.connect(self.close)
        
        # Enable buttons since engine is already loaded
        self.startButton.setEnabled(True)
        self.loadMapButton.setEnabled(True)
        self.statusLabel.setText("MATLAB engine ready")
    
    def loadMap(self):
        """Load CI map data from MATLAB."""
        # Check if MATLAB engine is ready
        if not hasattr(self, 'eng') or self.eng is None:
            QMessageBox.warning(self, "Error", "MATLAB engine is not ready. Please wait for initialization.")
            return
            
        name = self.nameInput.text().strip()
        surname = self.surnameInput.text().strip()
        side = self.sideCombo.currentText()
        mapNumber = self.mapNumberEdit.text()
        
        if not name or not surname:
            QMessageBox.warning(self, "Input Error", "Please enter both name and surname.")
            return
            
        if not mapNumber.isdigit():
            QMessageBox.warning(self, "Input Error", "Map number must be numeric.")
            return
        
        try:
            self.statusLabel.setText("Loading CI Map...")
            self.loadMapButton.setEnabled(False)  # Disable button while loading
            
            self.map_data = self.eng.call_map(name, surname, side, mapNumber, nargout=1)
            lower_levels = self.map_data['lower_levels']
            upper_levels = self.map_data['upper_levels']
            channel_stim_rate = self.map_data['channel_stim_rate_Hz']
            
            def clean_matlab_array(matlab_array):
                raw_str = str(matlab_array)
                cleaned = raw_str.replace('[', '').replace(']', '')
                values = []
                for item in cleaned.replace(',', ' ').split():
                    try:
                        values.append(f"{float(item):.1f}")
                    except ValueError:
                        pass
                return ", ".join(values)
            
            self.lowerLevelsLabel.setText(clean_matlab_array(lower_levels))
            self.upperLevelsLabel.setText(clean_matlab_array(upper_levels))
            self.channelStimRateLabel.setText(clean_matlab_array(channel_stim_rate))
            
            self.mapDataGroupLabel.setVisible(True)
            for i in range(self.mapDataLayout.count()):
                item = self.mapDataLayout.itemAt(i)
                if item.widget():
                    item.widget().setVisible(True)
            
            self.setFixedSize(1000, 800)
            self.center()
            
            self.startButton.setText("Start Experiment (Map Loaded)")
            self.startButton.setStyleSheet("background-color: #28a745;")
            self.mapDataGroupLabel.setText("CI Map Data")
            self.mapDataGroupLabel.setStyleSheet("color: #28a745;")
            
        except Exception as e:
            QMessageBox.critical(self, "MATLAB Error", str(e))
            self.map_data = None
        finally:
            self.loadMapButton.setEnabled(True)  # Re-enable button after loading
            self.statusLabel.setText("Ready")
    
    def startTest(self):
        """Start the selected test."""
        # Get input values
        name = self.nameInput.text()
        surname = self.surnameInput.text()
        id_val = self.idInput.text()
        test_type = self.testCombo.currentText()
        map_number = self.mapNumberEdit.text()
        map_side = self.sideCombo.currentText()

        # Check if map data is loaded; if not, show a styled warning and do not proceed
        if self.map_data is None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("No Map Loaded")
            msg.setText("No map is currently loaded.")
            msg.setInformativeText("Please load a map in the main window before starting the experiment.")
            msg.setStyleSheet("""
                QMessageBox { background-color: #2c3e50; }
                QMessageBox QLabel { color: white; font-size: 12px; padding: 10px; }
                QPushButton { background-color: #3498db; color: white; border: none; padding: 5px 15px; border-radius: 3px; min-width: 80px; }
                QPushButton:hover { background-color: #2980b9; }
            """)
            msg.exec_()
            return

        # Create and show test window
        if test_type == "HSM":
            self.test_window = HSMTestWindow(self.eng, name, surname, id_val, self.map_data, map_number, map_side)
        else:
            self.test_window = OLSATestWindow(self.eng, name, surname, id_val, self.map_data, map_number, map_side)
        
        # Connect window closing signal
        self.test_window.window_closing.connect(self.onTestWindowClosing)
        
        # Hide main window and show test window
        self.hide()
        self.test_window.show()


    
    def onTestWindowClosing(self):
        """Handle test window closing."""
        print("onTestWindowClosing called")
        # Stop the stream server and quit the application
        try:
            if hasattr(self, 'eng'):
                print("Stopping stream server...")
                self.eng.manageStreamServer('stop', nargout=0)
                print("Quitting MATLAB engine...")
                self.eng.quit()
        except Exception as e:
            print(f"Error stopping stream server: {e}")
        print("Quitting application...")
        QApplication.quit()
    
    def showHelp(self):
        """Show help information."""
        help_text = """
        Speech Intelligibility Experiment Help
        
        1. Enter subject information (name, surname, ID)
        2. Load CI Map data if available
        3. Select test type (HSM or OLSA)
        4. Click 'Start Test' to begin
        
        For more information, please contact the administrator.
        """
        QMessageBox.information(self, "Help", help_text)
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            if hasattr(self, 'eng'):
                self.eng.manageStreamServer('stop', nargout=0)
                self.eng.quit()
        except Exception as e:
            print(f"Error stopping stream server: {e}")
        event.accept()

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

def main():
    """Main function to start the application."""
    app = QApplication(sys.argv)
    
    # Initialize QSplashScreen
    splash_pix = QPixmap(640, 480)
    pastel_blue = QColor("#A7C7E7") 
    splash_pix.fill(pastel_blue)
    splash = QSplashScreen(splash_pix)
    splash.setWindowFlags(splash.windowFlags() | Qt.WindowStaysOnTopHint)
    
    # Center the text with larger font
    splash.setStyleSheet("color: black; font-size: 18px; font-weight: bold;")
    
    # Don't show the message yet - we'll position it relative to the GIF later
    
    # --- Add APG Logo to the right ---
    # Determine script directory explicitly before using it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Construct the path to the APG image
        apg_image_path = os.path.join(script_dir, "assets", "apg.png")
        
        if os.path.exists(apg_image_path):
            apg_pixmap = QPixmap(apg_image_path)
            if not apg_pixmap.isNull():
                
                # Scale to 30% of original size
                scale_factor = 0.3
                original_width = apg_pixmap.width()
                original_height = apg_pixmap.height()
                img_width = int(original_width * scale_factor)
                img_height = int(original_height * scale_factor)
                
                # Actually resize the pixmap
                scaled_pixmap = apg_pixmap.scaled(img_width, img_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # Create and position the label
                apg_label = QLabel(splash)
                apg_label.setPixmap(scaled_pixmap)
                
                # Position in the top right, with a margin
                margin = 20 
                splash_width = splash_pix.width()
                
                # Calculate position (top right corner)
                x_pos = splash_width - scaled_pixmap.width() - margin
                y_pos = margin  # Top margin
                
                apg_label.setGeometry(x_pos, y_pos, scaled_pixmap.width(), scaled_pixmap.height())
                apg_label.show()
            else:
                print(f"[WARN] APG logo file found but QPixmap is null: {apg_image_path}")
        else:
            # Fallback: Try looking directly in the current directory
            apg_image_path = os.path.join(script_dir, "apg.png")
            print(f"[DEBUG] Trying alternate path: {apg_image_path}")
            if os.path.exists(apg_image_path):
                print(f"[DEBUG] Found APG logo in alternate location: {apg_image_path}")
                # Now try to load from this alternate path
                apg_pixmap = QPixmap(apg_image_path)
                if not apg_pixmap.isNull():
                    print(f"[DEBUG] APG logo loaded from alternate path")
                    # Rest of the code to display the logo (same as above)
                    # ...
                else:
                    print(f"[WARN] APG logo file found at alternate path but QPixmap is null")
            else:
                print(f"[WARN] APG logo file not found at alternate path either")
    except Exception as e:
        print(f"[ERROR] Could not load or place APG logo: {e}")
        import traceback
        traceback.print_exc()
    # --- End Add APG Logo ---
    
    # Create a label for the animated GIF (loading indicator)
    animation_label = QLabel(splash)
    animation_label.setAttribute(Qt.WA_TranslucentBackground)  # Transparent background for label
    
    # Load the GIF from the assets folder
    gif_path = os.path.join(script_dir, "assets", "loading.gif")
    
    
    if not os.path.exists(gif_path):
        print(f"Error: GIF file not found at {gif_path}")
        # Show message centered if no GIF
        splash.showMessage("Loading...", Qt.AlignCenter, Qt.black)
    else:
        try:
            movie = QMovie(gif_path)
            if not movie.isValid():
                print(f"Error: QMovie could not validate the GIF file: {gif_path}")
                print(f"Supported formats: {QMovie.supportedFormats()}")
                # Show message centered if GIF invalid
                splash.showMessage("Loading...", Qt.AlignCenter, Qt.black)
            else:
                animation_label.setMovie(movie)
                movie.start()  # Start movie to obtain frame size
                original_size = movie.frameRect().size()
                
                if original_size.width() > 0 and original_size.height() > 0:
                    # Scale down the GIF (adjust scale_factor as desired)
                    scale_factor = 0.5 
                    new_width = max(1, int(original_size.width() * scale_factor))
                    new_height = max(1, int(original_size.height() * scale_factor))
                    movie.setScaledSize(QSize(new_width, new_height))
                    splash_width = splash_pix.width()
                    splash_height = splash_pix.height()
                    
                    # Center the GIF exactly
                    x = (splash_width - new_width) // 2
                    y = (splash_height - new_height) // 2 - 30  # Move up to make room for text
                    animation_label.setGeometry(x, y, new_width, new_height)
                    
                    # Position text below the GIF
                    text_y_pos = y + new_height + 10  # 10px gap between GIF and text
                    
                    # Create a custom text label for more precise positioning if needed
                    loading_text = QLabel("Loading...", splash)
                    loading_text.setStyleSheet("color: black; font-size: 18px; font-weight: bold; background: transparent;")
                    loading_text.setAlignment(Qt.AlignCenter)
                    text_width = 100  # Approximate width
                    text_x = (splash_width - text_width) // 2
                    loading_text.setGeometry(text_x, text_y_pos, text_width, 30)
                    loading_text.show()
                else:
                    print("Error: Original GIF size reported as 0x0 or invalid after starting.")
                    # Show message centered if GIF size is invalid
                    splash.showMessage("Loading...", Qt.AlignCenter, Qt.black)
                animation_label.show()
        except Exception as e:
            print(f"An unexpected error occurred while loading the GIF: {e}")
            # Show message centered if exception
            splash.showMessage("Loading...", Qt.AlignCenter, Qt.black)
    
    splash.show()
    app.processEvents()
    
    loader = MatlabEngineLoader()
    loader.engine_ready.connect(lambda engine: onEngineLoaded_animated(engine, splash))
    loader.error_occurred.connect(lambda msg: onEngineError(msg, app, splash))
    loader.start()
    
    sys.exit(app.exec_())

def onEngineLoaded_animated(engine, splash):
    """Handle MATLAB engine loaded with concurrent fade-out for the splash and fade-in for the main window."""
    def start_transition():
        global mainWindow
        try:
            mainWindow = MainWindow(engine)
        except Exception as e:
            onEngineError(f"Failed to create MainWindow: {e}", QApplication.instance(), splash)
            return

        # Set main window opacity to 0 (invisible) and show it immediately
        mainWindow.setWindowOpacity(0.0)
        mainWindow.show()  # Now the main window is present but transparent

        # Create fade-out animation for the splash screen
        splash_anim = QPropertyAnimation(splash, b"windowOpacity", splash)
        splash_anim.setDuration(700)  # Duration in milliseconds
        splash_anim.setStartValue(1.0)
        splash_anim.setEndValue(0.0)
        splash_anim.setEasingCurve(QEasingCurve.InOutQuad)
        # Once splash fade-out is done, close it
        splash_anim.finished.connect(lambda: splash.close())

        # Create fade-in animation for the main window
        fade_in = QPropertyAnimation(mainWindow, b"windowOpacity", mainWindow)
        fade_in.setDuration(700)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.InOutQuad)

        # Start both animations concurrently
        splash_anim.start(QPropertyAnimation.DeleteWhenStopped)
        fade_in.start(QPropertyAnimation.DeleteWhenStopped)

    QTimer.singleShot(0, start_transition)



def onEngineError(error_msg, app, splash):
    """Handle MATLAB engine loading error."""
    splash.hide()
    QMessageBox.critical(None, "Error", f"Failed to load MATLAB engine: {error_msg}")
    app.quit()

if __name__ == '__main__':
    main()
