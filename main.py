import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLineEdit, QLabel, 
                            QPushButton, QComboBox, QGridLayout, QVBoxLayout, QSpacerItem, 
                            QSizePolicy, QMessageBox, QSplashScreen, QFileDialog, QProgressBar, QHBoxLayout)
from PyQt5.QtGui import QFont, QPixmap, QMovie, QColor
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QRect, QEasingCurve

from core.matlab_engine import MatlabEngineLoader
from core.base_window import BaseWindow
from hsm.hsm_window import HSMTestWindow
from olsa.olsa_window import OLSATestWindow

class CustomSplashScreen(QSplashScreen):
    def mousePressEvent(self, event):
        # Ignore mouse press events to prevent the splash from closing
        event.ignore()

class MainWindow(BaseWindow):
    """Main window for the Speech Intelligibility Experiment."""
    
    def __init__(self, engine):
        super().__init__(None, "Speech Intelligibility Experiment")
        self.eng = engine
        self.map_data = None
        self.original_upper_levels = None # Stores the original matlab.double object
        self.original_upper_levels_numeric = None # Stores the original numeric list
        self.calibrated_upper_levels_numeric = None # Stores the calibrated numeric list
        self.calibration_percent = 80
        self.initUI()
        # Don't show the window immediately
        self.hide()
        self.setObjectName("mainWindow")  # Set object name for finding the window
        self.setFixedSize(1300, 750)  # Extra-wide window to ensure proper spacing
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
        
        # Create a container for Map Data and Calibration together
        mapAndCalibrationContainer = QHBoxLayout()
        mapAndCalibrationContainer.setSpacing(15)  # Add spacing between map data and calibration
        
        # LEFT SIDE: Map Data Section
        mapDataContainer = QVBoxLayout()
        mapDataContainer.setContentsMargins(0, 0, 0, 0)  # No margins to maximize space
        
        self.mapDataGroupLabel = QLabel("CI Map Data")
        self.mapDataGroupLabel.setFont(QFont("Arial", 16, QFont.Bold))
        self.mapDataGroupLabel.setStyleSheet("color: white;")
        mapDataContainer.addWidget(self.mapDataGroupLabel)
        
        self.mapDataLayout = QGridLayout()
        self.mapDataLayout.setVerticalSpacing(15)  # Reduced vertical spacing
        self.mapDataLayout.setHorizontalSpacing(20)  # Increase horizontal spacing between label and value
        
        # Create parameter labels with fixed width and right alignment
        thresholdLabel = QLabel("Thresholds (T Levels):")
        thresholdLabel.setStyleSheet("color: white;")
        thresholdLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        thresholdLabel.setFixedWidth(220)  # Fixed width ensures consistent alignment
        
        comfortableLabel = QLabel("Comfortable Levels (C Levels):")
        comfortableLabel.setStyleSheet("color: white;")
        comfortableLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        comfortableLabel.setFixedWidth(220)  # Fixed width ensures consistent alignment
        
        self.lowerLevelsLabel = QLabel("Not loaded")
        self.lowerLevelsLabel.setWordWrap(False)
        self.lowerLevelsLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lowerLevelsLabel.setStyleSheet("background-color: #2a2a2a; padding: 6px 10px; border-radius: 3px; color: white;")
        self.lowerLevelsLabel.setMinimumHeight(36)
        self.lowerLevelsLabel.setMinimumWidth(600)
        self.lowerLevelsLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Left align the values
        
        self.upperLevelsLabel = QLabel("Not loaded")
        self.upperLevelsLabel.setWordWrap(False)
        self.upperLevelsLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.upperLevelsLabel.setStyleSheet("background-color: #2a2a2a; padding: 6px 10px; border-radius: 3px; color: white;")
        self.upperLevelsLabel.setMinimumHeight(36)
        self.upperLevelsLabel.setMinimumWidth(600)
        self.upperLevelsLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Left align the values
        
        stimRateLabel = QLabel("Channel Stimulation Rate (Hz):")
        stimRateLabel.setStyleSheet("color: white;")
        stimRateLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        stimRateLabel.setFixedWidth(220)  # Fixed width ensures consistent alignment
        
        self.channelStimRateLabel = QLabel("Not loaded") 
        self.channelStimRateLabel.setWordWrap(False)
        self.channelStimRateLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.channelStimRateLabel.setStyleSheet("background-color: #2a2a2a; padding: 6px 10px; border-radius: 3px; color: white;")
        self.channelStimRateLabel.setMinimumHeight(36)
        self.channelStimRateLabel.setMinimumWidth(600)
        self.channelStimRateLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Left align the values
        
        # Use full row for each parameter to give maximum width
        self.mapDataLayout.addWidget(thresholdLabel, 0, 0, 1, 1, Qt.AlignVCenter)
        self.mapDataLayout.addWidget(self.lowerLevelsLabel, 0, 1, 1, 3, Qt.AlignVCenter)
        self.mapDataLayout.addWidget(comfortableLabel, 1, 0, 1, 1, Qt.AlignVCenter)
        self.mapDataLayout.addWidget(self.upperLevelsLabel, 1, 1, 1, 3, Qt.AlignVCenter)
        self.mapDataLayout.addWidget(stimRateLabel, 2, 0, 1, 1, Qt.AlignVCenter)
        self.mapDataLayout.addWidget(self.channelStimRateLabel, 2, 1, 1, 3, Qt.AlignVCenter)
        
        mapDataContainer.addLayout(self.mapDataLayout)
        
        # Add map data container to the left side
        mapAndCalibrationContainer.addLayout(mapDataContainer, 4)  # Map:Calibration ratio 4:1
        
        # RIGHT SIDE: Calibration Section with simplified direct widgets
        # Create a container widget with background for the calibration section - compact design
        self.calibrationWidget = QWidget()
        self.calibrationWidget.setStyleSheet("background-color: #333333; border: 2px solid #ff9900; border-radius: 8px; padding: 8px;")
        self.calibrationWidget.setFixedWidth(240)  # Slightly wider to fit content
        
        self.calibrationContainer = QVBoxLayout(self.calibrationWidget)
        self.calibrationContainer.setContentsMargins(8, 8, 8, 8)  # Minimal padding
        self.calibrationContainer.setSpacing(8)  # Minimal spacing
        
        # Label for the calibration section
        calibrationLabel = QLabel("Adjust Comfort Level:")
        calibrationLabel.setAlignment(Qt.AlignCenter)
        calibrationLabel.setStyleSheet("color: white; font-weight: bold;")
        self.calibrationContainer.addWidget(calibrationLabel)
        
        # Create a simple horizontal layout for controls
        controlsLayout = QHBoxLayout()
        controlsLayout.setSpacing(8)  # Reduced spacing
        controlsLayout.setContentsMargins(0, 0, 0, 0)  # No margins
        
        # Decrease button - more compact
        self.decreaseButton = QPushButton("–")
        self.decreaseButton.setFont(QFont("Arial", 16, QFont.Bold))
        self.decreaseButton.setFixedSize(40, 40)  # Compact size
        self.decreaseButton.setStyleSheet("background-color: #d35400; color: white;")
        self.decreaseButton.clicked.connect(self.decreaseCalibration)
        controlsLayout.addWidget(self.decreaseButton)
        
        # Percentage display - more compact
        self.calibrationPercentLabel = QLabel("80%")
        self.calibrationPercentLabel.setFont(QFont("Arial", 14, QFont.Bold))
        self.calibrationPercentLabel.setAlignment(Qt.AlignCenter)
        self.calibrationPercentLabel.setStyleSheet("color: #ffffff; background-color: #1e1e1e; padding: 6px; border-radius: 3px;")
        self.calibrationPercentLabel.setFixedWidth(70)
        controlsLayout.addWidget(self.calibrationPercentLabel)
        
        # Increase button - more compact
        self.increaseButton = QPushButton("+")
        self.increaseButton.setFont(QFont("Arial", 16, QFont.Bold))
        self.increaseButton.setFixedSize(40, 40)  # Compact size
        self.increaseButton.setStyleSheet("background-color: #27ae60; color: white;")
        self.increaseButton.clicked.connect(self.increaseCalibration)
        controlsLayout.addWidget(self.increaseButton)
        
        # Add controls layout to calibration container
        self.calibrationContainer.addLayout(controlsLayout)
        
        # Play button - directly below with no spacing
        self.playCalibrationButton = QPushButton("Play Test Sentence")
        self.playCalibrationButton.setMinimumHeight(40)  # Compact height
        self.playCalibrationButton.setStyleSheet("background-color: #ff9900; color: black; font-weight: bold; font-size: 13px;")
        self.playCalibrationButton.clicked.connect(self.playCalibration)
        self.calibrationContainer.addWidget(self.playCalibrationButton)
        
        # No stretch - compact layout with no empty space
        
        # Add the calibration container to the map and calibration container
        calibrationOuterContainer = QVBoxLayout()
        calibrationOuterContainer.setContentsMargins(30, 0, 0, 0)  # Increase left margin for better separation
        calibrationOuterContainer.addWidget(self.calibrationWidget)
        calibrationOuterContainer.addStretch(1)  # Keep the stretch here to push the widget to the top
        
        mapAndCalibrationContainer.addLayout(calibrationOuterContainer, 1)  # Map:Calibration ratio 3:1
        
        # Add the combined container to the main layout
        self.layout.addLayout(mapAndCalibrationContainer)
        
        # Hide map data section and calibration initially
        self.mapDataGroupLabel.setVisible(False)
        for i in range(self.mapDataLayout.count()):
            item = self.mapDataLayout.itemAt(i)
            if item.widget():
                item.widget().setVisible(False)
        
        # Hide calibration widget initially
        self.calibrationWidget.setVisible(False)
        
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
    
    def extract_values_from_matlab(self, matlab_array):
        """Extract numeric values from a MATLAB array string representation."""
        try:
            raw_str = str(matlab_array)
            # Remove brackets and split by spaces or commas
            cleaned = raw_str.replace('[', '').replace(']', '')
            values = []
            for item in cleaned.replace(',', ' ').split():
                try:
                    values.append(float(item))
                except ValueError:
                    pass
            return values
        except Exception as e:
            print(f"Error extracting values: {e}")
            return []
    
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
            
            # Extract and store original upper levels when loading map
            self.original_upper_levels = upper_levels
            # Also store the numeric representation of original levels
            self.original_upper_levels_numeric = self.extract_values_from_matlab(upper_levels)
            # Initialize calibrated values with the original ones
            self.calibrated_upper_levels_numeric = self.original_upper_levels_numeric[:]
            
            # Debug print
            print(f"Original upper levels (matlab): {self.original_upper_levels}")
            print(f"Original lower levels: {lower_levels}")
            print(f"Channel stim rate: {channel_stim_rate}")
            
            self.lowerLevelsLabel.setText(clean_matlab_array(lower_levels))
            self.upperLevelsLabel.setText(clean_matlab_array(upper_levels))
            self.channelStimRateLabel.setText(clean_matlab_array(channel_stim_rate))
            
            self.mapDataGroupLabel.setVisible(True)
            for i in range(self.mapDataLayout.count()):
                item = self.mapDataLayout.itemAt(i)
                if item.widget():
                    item.widget().setVisible(True)
            
            # Make sure window size is correct
            self.setFixedSize(1300, 750)  # Match the width in __init__
            self.center()
            
            self.startButton.setText("Start Experiment (Map Loaded)")
            self.startButton.setStyleSheet("background-color: #28a745;")
            self.mapDataGroupLabel.setText("CI Map Data")
            self.mapDataGroupLabel.setStyleSheet("color: #28a745;")
            
            # Force UI update to ensure map data is visible
            QApplication.processEvents()
            
            # Make sure the calibration controls are fully visible
            self.showCalibrationControls()
            
            # Apply calibration - this will update the upperLevelsLabel with calibrated values
            self.updateCalibration()
            
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
                /* font-size: 14px; Remove base font size */
            }
            /* Remove QLabel[class="ParameterValueLabel"] style */
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
            /* Remove QLineEdit style */
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

    def hideCalibrationControls(self):
        """Hide all calibration controls."""
        print("Hiding calibration controls")
        self.calibrationWidget.setVisible(False)
        
    def showCalibrationControls(self):
        """Show all calibration controls."""
        print("Showing calibration controls...")
        self.calibrationWidget.setVisible(True)
        # Force update
        QApplication.processEvents()

    def playCalibration(self):
        """Play a test sentence using current calibration settings."""
        print("Play calibration sentence button clicked")
        if not hasattr(self, 'map_data') or self.map_data is None:
            QMessageBox.warning(self, "Error", "No map data loaded. Please load a map first.")
            return
            
        try:
            QMessageBox.information(self, "Test Sentence", "Playing test sentence with current calibration settings.")
            # The actual implementation of playing the test sentence would go here
            # This would involve calling a MATLAB function
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error playing test sentence: {str(e)}")
            
    def increaseCalibration(self):
        """Increase calibration percentage."""
        print("Increase button clicked")
        if not hasattr(self, 'calibration_percent'):
            print("calibration_percent attribute not found")
            return
            
        # Increase by 5% up to a maximum of 120%
        self.calibration_percent = min(120, self.calibration_percent + 2)
        print(f"Calibration increased to {self.calibration_percent}%")
        self.updateCalibration()
        
    def decreaseCalibration(self):
        """Decrease calibration percentage."""
        print("Decrease button clicked")
        if not hasattr(self, 'calibration_percent'):
            print("calibration_percent attribute not found")
            return
            
        # Decrease by 5% down to a minimum of 20%
        self.calibration_percent = max(20, self.calibration_percent - 2)
        print(f"Calibration decreased to {self.calibration_percent}%")
        self.updateCalibration()

    def updateCalibration(self):
        """Update the calibrated levels based on the calibration percentage."""
        print(f"Updating calibration to {self.calibration_percent}%")
        # Use the stored numeric original levels for calculation
        if not hasattr(self, 'original_upper_levels_numeric') or not self.original_upper_levels_numeric:
            print("No original numeric upper levels data found for calibration")
            return
            
        try:
            # Update the percentage display
            self.calibrationPercentLabel.setText(f"{self.calibration_percent}%")
            
            # Use the stored numeric original values
            values = self.original_upper_levels_numeric
            print(f"Using original numeric values for calculation: {values}")
            
            if not values:
                print("No numeric values found in original_upper_levels_numeric")
                return
                
            # Calculate calibrated values
            calibrated_values = [val * (self.calibration_percent / 100.0) for val in values]
            # Store the numeric calibrated values
            self.calibrated_upper_levels_numeric = calibrated_values
            print(f"Stored numeric calibrated values: {[round(val, 1) for val in self.calibrated_upper_levels_numeric]}")
            
            # Format for display only
            formatted_values = [f"{val:.1f}" for val in self.calibrated_upper_levels_numeric]
            display_text = ", ".join(formatted_values)
            
            # Update display label
            self.upperLevelsLabel.setText(display_text)
            print(f"Updated display with: {display_text}")
            
            # DO NOT UPDATE self.map_data['upper_levels'] with the string anymore
            # Keep self.map_data['upper_levels'] as the original matlab object
            # if hasattr(self, 'map_data') and self.map_data is not None:
            #     calibrated_matlab_str = f"[{' '.join([f'{val:.6f}' for val in calibrated_values])}]"
            #     self.map_data['upper_levels'] = calibrated_matlab_str 
            #     print(f"Updated map_data upper_levels with calibrated values") # This line is removed
                
        except Exception as e:
            print(f"Error in updateCalibration: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to start the application."""
    app = QApplication(sys.argv)
    
    # Initialize QSplashScreen
    splash_pix = QPixmap(640, 480)
    pastel_blue = QColor("#A7C7E7") 
    splash_pix.fill(pastel_blue)
    splash = CustomSplashScreen(splash_pix)
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
