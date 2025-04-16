import os
import sys
import traceback
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QFont, QPixmap, QMovie, QColor, QPainter, QLinearGradient, QPen, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLineEdit, QLabel, QComboBox, QGridLayout, QVBoxLayout, QHBoxLayout,
    QSpacerItem, QSizePolicy, QMessageBox, QSplashScreen, QFileDialog, QProgressBar, QPushButton, QGraphicsDropShadowEffect,
    QFrame, QTextEdit, QGroupBox, QCheckBox
)

import matlab

# Import your existing modules (these are assumed to exist as in your original code)
from core.matlab_engine import MatlabEngineLoader
from core.base_window import BaseWindow
from core.ui_components import CustomMessageBox
from hsm.hsm_window import HSMTestWindow
from olsa.olsa_window import OLSATestWindow

import torchaudio

from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QPushButton, QGraphicsDropShadowEffect

class CoolPushButton(QPushButton):
    """A QPushButton subclass that adds a smoothly animated drop-shadow effect on hover."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        
        # Apply a custom stylesheet for a modern look.
        self.setStyleSheet("""
        CoolPushButton {
            border: none;
            background-color: #0078D7;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 14px;
        }
        CoolPushButton:hover {
            background-color: #005EA2;
        }
        CoolPushButton:pressed {
            background-color: #004371;
        }
        """)
        # Change the cursor to a pointer to indicate a clickable button.
        self.setCursor(Qt.PointingHandCursor)
        
        # Create and attach the shadow effect with an initial blur radius of 0.
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(0)
        self._shadow.setOffset(0, 0)
        # Use a subtle dark shadow for a more professional effect.
        self._shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(self._shadow)
        
        # Setup the property animation for the blurRadius property.
        self._animation = QPropertyAnimation(self._shadow, b"blurRadius", self)
        self._animation.setDuration(150)  # Duration in milliseconds.
        self._animation.setEasingCurve(QEasingCurve.InOutCubic)
    
    def enterEvent(self, event):
        # Animate to a blur radius of 15 when the mouse enters.
        self._animation.stop()
        self._animation.setStartValue(self._shadow.blurRadius())
        self._animation.setEndValue(15)
        self._animation.start()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        # Animate back to a blur radius of 0 when the mouse leaves.
        self._animation.stop()
        self._animation.setStartValue(self._shadow.blurRadius())
        self._animation.setEndValue(0)
        self._animation.start()
        super().leaveEvent(event)


class CustomSplashScreen(QSplashScreen):
    def mousePressEvent(self, event):
        # Ignore mouse press events to prevent the splash from closing
        event.ignore()


class ElectrodeVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)  # Increased height for better spacing
        self.t_levels = []
        self.c_levels = []
        self.stim_rate = 0
        self.electrodes_available = []
        self.setStyleSheet("background-color: #1a1a1a; border-radius: 5px;")
        
    def update_levels(self, t_levels, c_levels, electrodes_available=None, stim_rate=None):
        self.t_levels = t_levels
        self.c_levels = c_levels
        if electrodes_available is not None:
            try:
                # Handle MATLAB array properly
                if hasattr(electrodes_available, '_data'):  # If it's a MATLAB array
                    self.electrodes_available = [int(x) for x in electrodes_available._data]
                else:  # If it's already a list or other format
                    raw_str = str(electrodes_available).replace("[", "").replace("]", "")
                    self.electrodes_available = [int(float(x)) for x in raw_str.split() if x.strip()]
            except Exception as e:
                print(f"Error converting electrodes_available: {e}")
                self.electrodes_available = []
        if stim_rate is not None:
            self.stim_rate = stim_rate
        self.update()
        
    def paintEvent(self, event):
        if not self.t_levels or not self.c_levels:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Setup fonts
        number_font = QFont("Arial", 9)
        value_font = QFont("Arial", 9)
        
        # Calculate dimensions
        width = self.width()
        height = self.height()
        top_margin = 35  # Space for legend
        bottom_margin = 50  # Increased space for electrode numbers
        left_margin = 50  # Increased space for scale numbers
        right_margin = 10
        
        bar_height = height - top_margin - bottom_margin
        usable_width = width - left_margin - right_margin
        electrode_width = usable_width / len(self.t_levels)
        bar_width = max(12, int(electrode_width * 0.7))  # Slightly wider bars
        
        # Draw legend with stim rate
        legend_y = 10
        painter.setFont(value_font)
        painter.setPen(QPen(QColor("#ffffff")))
        
        # T-Level legend
        painter.fillRect(left_margin, legend_y, 15, 10, QColor("#2196F3"))
        painter.drawText(left_margin + 20, legend_y + 10, "T-Levels")
        
        # C-Level legend
        painter.fillRect(left_margin + 120, legend_y, 15, 10, QColor("#FF9800"))
        painter.drawText(left_margin + 140, legend_y + 10, "C-Levels")
        
        # Stim rate on the right
        painter.setPen(QPen(QColor("#4CAF50")))
        stim_text = f"Stim Rate: {self.stim_rate:.0f} Hz"
        painter.drawText(width - 150, legend_y + 10, stim_text)
        
        # Draw scale on the left (steps of 50)
        painter.setFont(value_font)
        painter.setPen(QPen(QColor("#666666")))
        for i in range(6):  # 0 to 250 in steps of 50
            y = top_margin + (bar_height * i // 5)
            level = 250 - (i * 50)  # Scale: 250, 200, 150, 100, 50, 0
            
            # Draw subtle horizontal grid line
            painter.drawLine(left_margin, y, width - right_margin, y)
            
            # Draw scale number with more space from grid
            painter.drawText(2, y - 5, left_margin - 8, 10, Qt.AlignRight | Qt.AlignVCenter, str(level))
        
        # Draw electrode bars and numbers
        painter.setFont(number_font)
        for i in range(len(self.t_levels)):
            x = int(left_margin + i * electrode_width + (electrode_width - bar_width) / 2)
            
            # Draw T-Level bar (blue)
            t_height = int((self.t_levels[i] / 255) * bar_height)
            t_y = height - bottom_margin - t_height
            painter.fillRect(x, t_y, bar_width, t_height, QColor("#2196F3"))
            
            # Draw T-Level value inside the blue bar in white
            t_text = f"{self.t_levels[i]:.0f}"
            t_text_rect = QRect(x, t_y + t_height//2 - 10, bar_width, 20)
            painter.setPen(QPen(QColor("#ffffff")))  # White text
            painter.drawText(t_text_rect, Qt.AlignCenter, t_text)
            
            # Draw C-Level marker (orange)
            c_height = int((self.c_levels[i] / 255) * bar_height)
            c_y = height - bottom_margin - c_height
            painter.fillRect(x, c_y - 2, bar_width, 4, QColor("#FF9800"))
            
            # Draw C-Level value above the marker
            c_text = f"{self.c_levels[i]:.0f}"
            c_text_rect = QRect(x, c_y - 20, bar_width, 20)
            painter.setPen(QPen(QColor("#FF9800")))
            painter.drawText(c_text_rect, Qt.AlignCenter, c_text)
            
            # Draw electrode number with more space from bottom
            number_rect = QRect(x, height - 35, bar_width, 20)  # Moved up from bottom
            painter.setPen(QPen(QColor("#888888")))
            # Use available electrode number if present, otherwise use index + 1
            electrode_number = str(self.electrodes_available[i]) if self.electrodes_available else str(i + 1)
            painter.drawText(number_rect, Qt.AlignCenter, electrode_number)

class ParameterSection(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 1em;
                padding: 5px;
                color: #ececec;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)


class MainWindow(BaseWindow):
    """Main window for the Speech Intelligibility Experiment."""
    CALIBRATION_STEP = 2  # Calibration step percentage
    CALIBRATION_MIN = 20
    CALIBRATION_MAX = 120

    def __init__(self, engine):
        super().__init__(None, "Speech Intelligibility Experiment")
        self.eng = engine
        self.map_data = None
        self.original_upper_levels = None  # Original MATLAB object
        self.original_upper_levels_numeric = None  # Numeric list version
        self.calibrated_upper_levels_numeric = None  # Calibrated numeric list
        self.calibration_percent = 80
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        self.initUI()
        self.hide()  # Do not show immediately
        self.setObjectName("mainWindow")
        self.setFixedSize(1300, 750)
        self.center()
        self.applyStyles()

    def initUI(self):
        """Initialize and build the GUI elements."""
        super().initUI()

        # CI Streaming Checkbox (absolute positioned)
        self.streamingCheckbox = QCheckBox("CI Streaming")
        self.streamingCheckbox.setToolTip("Enable/Disable CI Streaming")
        self.streamingCheckbox.setStyleSheet("""
            QCheckBox {
                color: #95a5a6;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border: 1px solid #3498db;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                background: #2ecc71;
            }
        """)
        self.streamingCheckbox.setChecked(True)
        self.streamingCheckbox.setParent(self)
        self.streamingCheckbox.move(self.width() - 100, 15)
        
        # Ensure checkbox stays in correct position when window resizes
        def updateCheckboxPosition():
            self.streamingCheckbox.move(self.width() - 100, 15)
            self.streamingCheckbox.raise_()
        self.resizeEvent = lambda e: updateCheckboxPosition()

        # Title label
        titleLabel = QLabel("HSM Speech Intelligibility Experiment")
        titleLabel.setFont(QFont("Arial", 22, QFont.Bold))
        titleLabel.setAlignment(Qt.AlignCenter)
        titleLabel.setStyleSheet("color: #ececec;")
        self.layout.addWidget(titleLabel)

        # Subject Information Section
        subjectGroupLabel = QLabel("Subject Information")
        subjectGroupLabel.setFont(QFont("Arial", 16, QFont.Bold))
        subjectGroupLabel.setStyleSheet("color: #ececec;")
        self.layout.addWidget(subjectGroupLabel)

        subjectGrid = QGridLayout()
        subjectGrid.setSpacing(10)
        subjectGrid.setAlignment(Qt.AlignLeft)  # Align the entire grid to the left

        nameLabel = QLabel("Name:")
        nameLabel.setStyleSheet("color: #ececec;")
        self.nameInput = QLineEdit("wolf-dieter")
        self.nameInput.setFixedWidth(150)

        surnameLabel = QLabel("Surname:")
        surnameLabel.setStyleSheet("color: #ececec;")
        self.surnameInput = QLineEdit("Goecke")
        self.surnameInput.setFixedWidth(150)

        idLabel = QLabel("ID:")
        idLabel.setStyleSheet("color: #ececec;")
        self.idInput = QLineEdit("12345")
        self.idInput.setFixedWidth(150)


        # Add widgets to grid with proper alignment
        subjectGrid.addWidget(nameLabel, 0, 0, Qt.AlignLeft)
        subjectGrid.addWidget(self.nameInput, 0, 1, Qt.AlignLeft)
        subjectGrid.addWidget(surnameLabel, 1, 0, Qt.AlignLeft)
        subjectGrid.addWidget(self.surnameInput, 1, 1, Qt.AlignLeft)
        subjectGrid.addWidget(idLabel, 2, 0, Qt.AlignLeft)
        subjectGrid.addWidget(self.idInput, 2, 1, Qt.AlignLeft)

        self.layout.addLayout(subjectGrid)

        # Separator
        separator1 = QLabel()
        separator1.setFixedHeight(1)
        separator1.setStyleSheet("background-color: #555;")
        self.layout.addWidget(separator1)

        # Map Settings Section
        mapGroupLabel = QLabel("CI Map Settings")
        mapGroupLabel.setFont(QFont("Arial", 16, QFont.Bold))
        mapGroupLabel.setStyleSheet("color: #ececec;")
        self.layout.addWidget(mapGroupLabel)

        mapGrid = QGridLayout()
        mapGrid.setSpacing(10)

        mapSideLabel = QLabel("Hearing Side:")
        mapSideLabel.setStyleSheet("color: #ececec;")
        self.sideCombo = QComboBox()
        self.sideCombo.addItems(["Left", "Right"])
        self.sideCombo.setFixedWidth(150)

        mapNumberLabel = QLabel("Map Number:")
        mapNumberLabel.setStyleSheet("color: #ececec;")
        self.mapNumberEdit = QLineEdit("34")

        mapGrid.addWidget(mapSideLabel, 0, 0)
        mapGrid.addWidget(self.sideCombo, 0, 1)
        mapGrid.addWidget(mapNumberLabel, 1, 0)
        mapGrid.addWidget(self.mapNumberEdit, 1, 1)
        self.layout.addLayout(mapGrid)

        self.loadMapButton = CoolPushButton("Load Map")
        self.loadMapButton.clicked.connect(self.loadMap)
        self.layout.addWidget(self.loadMapButton)

        # Separator
        separator2 = QLabel()
        separator2.setFixedHeight(1)
        separator2.setStyleSheet("background-color: #555;")
        self.layout.addWidget(separator2)

        # Container for Map Data and Calibration
        mapAndCalibrationContainer = QHBoxLayout()
        mapAndCalibrationContainer.setSpacing(15)

        # LEFT SIDE: Map Data Section
        mapDataContainer = QVBoxLayout()
        mapDataContainer.setContentsMargins(0, 0, 0, 0)

        self.mapDataGroupLabel = QLabel("CI Map Data")
        self.mapDataGroupLabel.setFont(QFont("Arial", 16, QFont.Bold))
        self.mapDataGroupLabel.setStyleSheet("color: #ececec;")
        mapDataContainer.addWidget(self.mapDataGroupLabel)

        # Create electrode visualizer
        self.electrode_visualizer = ElectrodeVisualizer()
        self.electrode_visualizer.setMinimumHeight(150)
        mapDataContainer.addWidget(self.electrode_visualizer)
        
        # T-Levels Section
        self.t_levels_section = ParameterSection("Threshold Levels (T-Levels)")
        self.t_levels_text = QTextEdit()
        self.t_levels_text.setReadOnly(True)
        self.t_levels_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #2196F3;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        self.t_levels_text.setMaximumHeight(100)
        self.t_levels_section.layout.addWidget(self.t_levels_text)
        mapDataContainer.addWidget(self.t_levels_section)
        
        # C-Levels Section
        self.c_levels_section = ParameterSection("Comfort Levels (C-Levels)")
        self.c_levels_text = QTextEdit()
        self.c_levels_text.setReadOnly(True)
        self.c_levels_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #FF9800;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        self.c_levels_text.setMaximumHeight(100)
        self.c_levels_section.layout.addWidget(self.c_levels_text)
        mapDataContainer.addWidget(self.c_levels_section)
        
        # Stimulation Parameters Section
        self.stim_params_section = ParameterSection("Stimulation Parameters")
        self.stim_rate_text = QTextEdit()
        self.stim_rate_text.setReadOnly(True)
        self.stim_rate_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #4CAF50;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        self.stim_rate_text.setMaximumHeight(60)
        self.stim_params_section.layout.addWidget(self.stim_rate_text)
        mapDataContainer.addWidget(self.stim_params_section)

        mapAndCalibrationContainer.addLayout(mapDataContainer, 4)  # Ratio 4:1 for Map Data

        # RIGHT SIDE: Calibration Section
        self.calibrationWidget = QWidget()
        self.calibrationWidget.setStyleSheet(
            "background-color: #2c3e50; border-radius: 8px; padding: 5px;"
        )
        self.calibrationWidget.setFixedWidth(240)

        self.calibrationContainer = QVBoxLayout(self.calibrationWidget)
        self.calibrationContainer.setContentsMargins(8, 8, 8, 8)
        self.calibrationContainer.setSpacing(8)

        calibrationLabel = QLabel("Adjust Comfort Level:")
        calibrationLabel.setAlignment(Qt.AlignCenter)
        calibrationLabel.setStyleSheet("color: #ececec; font-weight: bold;")
        self.calibrationContainer.addWidget(calibrationLabel)

        controlsLayout = QHBoxLayout()
        controlsLayout.setSpacing(8)
        controlsLayout.setContentsMargins(0, 0, 0, 0)

        self.decreaseButton = CoolPushButton("–")
        self.decreaseButton.setFont(QFont("Arial", 16, QFont.Bold))
        self.decreaseButton.setFixedSize(40, 40)
        self.decreaseButton.setStyleSheet("background-color: #d35400; color: white; border-radius: 20px;")
        self.decreaseButton.clicked.connect(self.decreaseCalibration)
        controlsLayout.addWidget(self.decreaseButton)

        self.calibrationPercentLabel = QLabel(f"{self.calibration_percent}%")
        self.calibrationPercentLabel.setFont(QFont("Arial", 14, QFont.Bold))
        self.calibrationPercentLabel.setAlignment(Qt.AlignCenter)
        self.calibrationPercentLabel.setStyleSheet(
            "color: #ffffff; background-color: #1e2b38; padding: 6px; border-radius: 3px;"
        )
        self.calibrationPercentLabel.setFixedWidth(70)
        controlsLayout.addWidget(self.calibrationPercentLabel)

        self.increaseButton = CoolPushButton("+")
        self.increaseButton.setFont(QFont("Arial", 16, QFont.Bold))
        self.increaseButton.setFixedSize(40, 40)
        self.increaseButton.setStyleSheet("background-color: #27ae60; color: white; border-radius: 20px;")
        self.increaseButton.clicked.connect(self.increaseCalibration)
        controlsLayout.addWidget(self.increaseButton)

        self.calibrationContainer.addLayout(controlsLayout)

        self.playCalibrationButton = CoolPushButton("Play Test Sentence")
        self.playCalibrationButton.setMinimumHeight(40)
        self.playCalibrationButton.setStyleSheet(
            "background-color: #ff9900; color: black; font-weight: bold; font-size: 13px; border-radius: 8px;"
        )
        self.playCalibrationButton.clicked.connect(self.playCalibration)
        self.calibrationContainer.addWidget(self.playCalibrationButton)

        calibrationOuterContainer = QVBoxLayout()
        calibrationOuterContainer.setContentsMargins(30, 0, 0, 0)
        calibrationOuterContainer.addWidget(self.calibrationWidget)
        calibrationOuterContainer.addStretch(1)

        mapAndCalibrationContainer.addLayout(calibrationOuterContainer, 1)
        self.layout.addLayout(mapAndCalibrationContainer)

        # Initially hide map data and calibration controls
        self.mapDataGroupLabel.setVisible(False)
        self.electrode_visualizer.setVisible(False)
        self.t_levels_section.setVisible(False)
        self.c_levels_section.setVisible(False)
        self.stim_params_section.setVisible(False)
        self.calibrationWidget.setVisible(False)

        # Add spacing between map data and bottom buttons
        self.layout.addSpacing(30)

        # Action Buttons
        btnLayout = QGridLayout()
        btnLayout.setSpacing(20)

        self.startButton = QPushButton("Start Experiment")
        self.startButton.setEnabled(False)
        self.startButton.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:enabled {
                background-color: #2ecc71;
            }
            QPushButton:enabled:hover {
                background-color: #27ae60;
            }
            QPushButton:enabled:pressed {
                background-color: #219a52;
            }
        """)
        self.helpButton = CoolPushButton("Help")
        self.exitButton = CoolPushButton("Exit")

        btnLayout.addWidget(self.startButton, 0, 0)
        btnLayout.addWidget(self.helpButton, 0, 1)
        btnLayout.addWidget(self.exitButton, 0, 2)
        self.layout.addLayout(btnLayout)

        # Connect signals
        self.startButton.clicked.connect(self.startTest)
        self.helpButton.clicked.connect(self.showHelp)
        self.exitButton.clicked.connect(self.close)

        # Initialize button states
        self.startButton.setEnabled(False)
        self.loadMapButton.setEnabled(True)
        self.statusLabel.setText("MATLAB engine ready")

    @staticmethod
    def _clean_matlab_array(matlab_array):
        """Helper method to extract and format numeric values from a MATLAB array."""
        try:
            raw_str = str(matlab_array).replace("[", "").replace("]", "")
            values = [
                f"{float(item):.1f}"
                for item in raw_str.replace(",", " ").split()
                if item.strip()
            ]
            return ", ".join(values)
        except Exception as e:
            print(f"Error cleaning MATLAB array: {e}")
            return ""

    def extract_values_from_matlab(self, matlab_array):
        """Extract numeric values from a MATLAB array string representation."""
        try:
            raw_str = str(matlab_array).replace("[", "").replace("]", "")
            values = []
            for item in raw_str.replace(",", " ").split():
                try:
                    values.append(float(item))
                except ValueError:
                    continue
            return values
        except Exception as e:
            print(f"Error extracting values: {e}")
            return []

    def loadMap(self):
        """Load CI map data from MATLAB."""
        if not hasattr(self, "eng") or self.eng is None:
            msg = CustomMessageBox(
                self,
                "Error",
                "MATLAB engine is not ready. Please wait for initialization.",
                QMessageBox.Warning
            )
            msg.exec_()
            return

        name = self.nameInput.text().strip()
        surname = self.surnameInput.text().strip()
        side = self.sideCombo.currentText()
        mapNumber = self.mapNumberEdit.text()

        if not name or not surname:
            msg = CustomMessageBox(
                self,
                "Input Error",
                "Please enter both name and surname.",
                QMessageBox.Warning
            )
            msg.exec_()
            return

        if not mapNumber.isdigit():
            msg = CustomMessageBox(
                self,
                "Input Error",
                "Map number must be numeric.",
                QMessageBox.Warning
            )
            msg.exec_()
            return
            
        # Add validation for map number > 0
        if int(mapNumber) <= 0:
            msg = CustomMessageBox(
                self,
                "Input Error",
                "Map number must be greater than 0.",
                QMessageBox.Warning
            )
            msg.exec_()
            return

        try:
            self.statusLabel.setText("Loading CI Map...")
            self.loadMapButton.setEnabled(False)

            self.map_data = self.eng.call_map(name, surname, mapNumber, nargout=1)
            lower_levels = self.map_data["lower_levels"]
            upper_levels = self.map_data["upper_levels"]
            channel_stim_rate = self.map_data["channel_stim_rate"]
            electrodes_available = self.map_data["electrodes"]

            # Store original upper levels and prepare numeric versions
            self.original_upper_levels = upper_levels
            self.original_upper_levels_numeric = self.extract_values_from_matlab(upper_levels)
            
            # Always start at 80% calibration
            self.calibration_percent = 80
            self.calibrated_upper_levels_numeric = [level * 0.8 for level in self.original_upper_levels_numeric]
            
            # Update the electrode visualizer with all data
            t_levels = self.extract_values_from_matlab(lower_levels)
            c_levels = self.calibrated_upper_levels_numeric  # Use calibrated levels instead of original
            self.map_data["comfort_levels"] = c_levels
            stim_rate = self.extract_values_from_matlab(channel_stim_rate)[0]
            self.electrode_visualizer.update_levels(t_levels, c_levels, electrodes_available, stim_rate)

            # Show only the electrode visualizer
            self.mapDataGroupLabel.setVisible(True)
            self.electrode_visualizer.setVisible(True)
            
            # Hide the text sections
            self.t_levels_section.setVisible(False)
            self.c_levels_section.setVisible(False)
            self.stim_params_section.setVisible(False)

            self.setFixedSize(1300, 750)
            self.center()

            self.startButton.setText("Start Experiment")
            self.startButton.setStyleSheet("background-color: #28a745;")
            self.mapDataGroupLabel.setText("CI Map Data")
            self.mapDataGroupLabel.setStyleSheet("color: #28a745;")
            QApplication.processEvents()

            self.showCalibrationControls()
            # Update calibration display
            self.calibrationPercentLabel.setText("80%")
            
            # Update button state without changing text
            self.startButton.setEnabled(True)
            
        except Exception as e:
            # If loading fails, ensure button is disabled
            self.startButton.setEnabled(False)
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.loadMapButton.setEnabled(True)
            self.statusLabel.setText("Ready")

    def _format_level_data(self, level_type, data):
        """Format level data for display with channel numbers."""
        values = self.extract_values_from_matlab(data)
        formatted = f"{level_type}:\n\n"
        for i, value in enumerate(values):
            formatted += f"Channel {i+1}: {value:.1f}\n"
        return formatted

    def _format_stim_data(self, stim_rate):
        """Format stimulation rate data for display."""
        rate = self.extract_values_from_matlab(stim_rate)[0]
        return f"Channel Stimulation Rate: {rate:.1f} Hz\n"

    def startTest(self):
        """Start the selected test."""


        name = self.nameInput.text()
        surname = self.surnameInput.text()
        id_val = self.idInput.text()
        test_type = 'HSM'
        map_number = self.mapNumberEdit.text()
        map_side = self.sideCombo.currentText()

        if self.map_data is None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("No Map Loaded")
            msg.setText("No map is currently loaded.")
            msg.setInformativeText(
                "Please load a map in the main window before starting the experiment."
            )
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QMessageBox QLabel {
                    color: black;
                    font-size: 12px;
                }
                QPushButton {
                    padding: 5px 15px;
                }
            """)
            msg.exec_()
            return

        if test_type == "HSM":
            self.test_window = HSMTestWindow(
                self.eng, name, surname, id_val, self.map_data, map_number, map_side,
                ci_streaming_enabled=self.streamingCheckbox.isChecked()
            )
        else:
            self.test_window = OLSATestWindow(
                self.eng, name, surname, id_val, self.map_data, map_number, map_side,
                ci_streaming_enabled=self.streamingCheckbox.isChecked()
            )

        self.test_window.window_closing.connect(self.onTestWindowClosing)
        self.hide()
        self.test_window.show()

    def onTestWindowClosing(self):
        """Handle test window closing."""
        try:
            if hasattr(self, "eng"):
                self.eng.manageStreamServer("stop", nargout=0)
                self.eng.quit()
        except Exception as e:
            print(f"Error stopping stream server: {e}")
        QApplication.quit()

    def showHelp(self):
        """Show help information."""
        help_text = (
            "Speech Intelligibility Experiment Help\n\n"
            "1. Enter subject information (name, surname, ID)\n"
            "2. Load CI Map data if available\n"
            "3. Select test type (HSM or OLSA)\n"
            "4. Click 'Start Test' to begin\n\n"
            "For more information, please contact the administrator."
        )
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Help")
        msg.setText(help_text)
        msg.setStyleSheet("""
            QMessageBox QLabel {
                color: black;
                font-size: 12px;
            }
        """)
        msg.exec_()

    def closeEvent(self, event):
        """Handle window close event."""
        try:
            if hasattr(self, "eng"):
                self.eng.manageStreamServer("stop", nargout=0)
                self.eng.quit()
        except Exception as e:
            print(f"Error stopping stream server: {e}")
        event.accept()

    def applyStyles(self):
        """Apply modern styles to the window."""
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #2c3e50;
            }
            QLabel {
                color: #ececec;
            }
            QLineEdit {
                background-color: #34495e;
                color: #ececec;
                border: 1px solid #3498db;
                padding: 5px;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2472a4;
            }
            QComboBox {
                background-color: #34495e;
                color: #ececec;
                border: 1px solid #3498db;
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
                border: 1px solid #3498db;
                border-radius: 4px;
                text-align: center;
                background-color: #34495e;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 4px;
            }
            """
        )

    def hideCalibrationControls(self):
        """Hide calibration controls."""
        self.calibrationWidget.setVisible(False)

    def showCalibrationControls(self):
        """Show calibration controls."""
        self.calibrationWidget.setVisible(True)
        QApplication.processEvents()

    def playCalibration(self):
        """Play a test sentence using current calibration settings."""
        if not self.map_data:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("No map data loaded. Please load a map first.")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QMessageBox QLabel {
                    color: black;
                    font-size: 12px;
                }
                QPushButton {
                    padding: 5px 15px;
                }
            """)
            msg.exec_()
            return

        try:
            speech_tensor, fs_speech = torchaudio.load('hsm/audio/speech/HSM_576_SNR=10_CCITT_clean.wav')
            
            # Convert to numpy array
            speech_data = speech_tensor.numpy()
            
            # Check for stereo files and convert to mono if needed
            if len(speech_data.shape) > 1 and speech_data.shape[0] > 1:
                speech_data = speech_data[0, :]  # Take first channel
                
            # Convert to MATLAB-compatible numeric array
            speech_data_matlab = matlab.double(speech_data.tolist())
            self.eng.stream(self.map_data, speech_data_matlab, self.streamingCheckbox.isChecked(), nargout=0)
        except Exception as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Error playing test sentence: {e}")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QMessageBox QLabel {
                    color: black;
                    font-size: 12px;
                }
                QPushButton {
                    padding: 5px 15px;
                }
            """)
            msg.exec_()

    def increaseCalibration(self):
        """Increase calibration percentage."""
        self.calibration_percent = min(
            self.CALIBRATION_MAX, self.calibration_percent + self.CALIBRATION_STEP
        )
        self.updateCalibration()

    def decreaseCalibration(self):
        """Decrease calibration percentage."""
        self.calibration_percent = max(
            self.CALIBRATION_MIN, self.calibration_percent - self.CALIBRATION_STEP
        )
        self.updateCalibration()

    def updateCalibration(self):
        """Update calibration and refresh displays."""
        if not self.original_upper_levels_numeric:
            return
            
        # Update calibrated levels
        self.calibrated_upper_levels_numeric = [
            level * (self.calibration_percent / 100)
            for level in self.original_upper_levels_numeric
        ]
        
        # Update the map_data with calibrated levels
        self.map_data["upper_levels"] = matlab.double(self.calibrated_upper_levels_numeric)
        self.map_data["comfort_levels"] = matlab.double(self.calibrated_upper_levels_numeric)
        
        # Update the electrode visualizer with new C-levels
        t_levels = self.extract_values_from_matlab(self.map_data["lower_levels"])
        self.electrode_visualizer.update_levels(t_levels, self.calibrated_upper_levels_numeric)
        
        # Update calibration percentage label
        self.calibrationPercentLabel.setText(f"{self.calibration_percent}%")

    def showError(self, title, message):
        """Show error message box."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: black;
                font-size: 12px;
            }
            QPushButton {
                padding: 5px 15px;
            }
        """)
        msg.exec_()


def main():
    app = QApplication(sys.argv)

    # Create a splash pixmap with a solid pastel blue background
    splash_width, splash_height = 640, 480
    splash_pix = QPixmap(splash_width, splash_height)
    splash_pix.fill(QColor("#e3f2fd"))  # Soft pastel blue
    
    splash = CustomSplashScreen(splash_pix)
    splash.setWindowFlags(splash.windowFlags() | Qt.WindowStaysOnTopHint)
    splash.setStyleSheet("color: black; font-size: 18px; font-weight: bold;")  # Changed text color to black

    # Add APG Logo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        apg_image_path = os.path.join(script_dir, "assets", "apg.png")
        if os.path.exists(apg_image_path):
            apg_pixmap = QPixmap(apg_image_path)
            if not apg_pixmap.isNull():
                scale_factor = 0.3
                img_width = int(apg_pixmap.width() * scale_factor)
                img_height = int(apg_pixmap.height() * scale_factor)
                scaled_pixmap = apg_pixmap.scaled(img_width, img_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                apg_label = QLabel(splash)
                apg_label.setPixmap(scaled_pixmap)
                margin = 20
                splash_width = splash_pix.width()
                x_pos = splash_width - scaled_pixmap.width() - margin
                y_pos = margin
                apg_label.setGeometry(x_pos, y_pos, scaled_pixmap.width(), scaled_pixmap.height())
                apg_label.show()
            else:
                print(f"[WARN] APG logo file found but QPixmap is null: {apg_image_path}")
        else:
            alt_path = os.path.join(script_dir, "apg.png")
            print(f"[DEBUG] Trying alternate path: {alt_path}")
            if os.path.exists(alt_path):
                apg_pixmap = QPixmap(alt_path)
                if not apg_pixmap.isNull():
                    # Additional code to handle the alternate path can be inserted here
                    pass
                else:
                    print("[WARN] APG logo file found at alternate path but QPixmap is null")
            else:
                print("[WARN] APG logo file not found at alternate path either")
    except Exception as e:
        print(f"[ERROR] Could not load or place APG logo: {e}")
        traceback.print_exc()

    # Animated GIF for loading indicator
    animation_label = QLabel(splash)
    animation_label.setAttribute(Qt.WA_TranslucentBackground)
    gif_path = os.path.join(script_dir, "assets", "loading3.gif")
    if not os.path.exists(gif_path):
        print(f"Error: GIF file not found at {gif_path}")
        splash.showMessage("Loading...", Qt.AlignCenter)
    else:
        try:
            movie = QMovie(gif_path)
            if not movie.isValid():
                print(f"Error: QMovie could not validate the GIF file: {gif_path}")
                splash.showMessage("Loading...", Qt.AlignCenter, Qt.black)
            else:
                animation_label.setMovie(movie)
                movie.start()
                original_size = movie.frameRect().size()
                if original_size.width() > 0 and original_size.height() > 0:
                    scale_factor = 0.5
                    new_width = max(1, int(original_size.width() * scale_factor))
                    new_height = max(1, int(original_size.height() * scale_factor))
                    movie.setScaledSize(QSize(new_width, new_height))
                    splash_width = splash_pix.width()
                    splash_height = splash_pix.height()
                    x = (splash_width - new_width) // 2
                    y = (splash_height - new_height) // 2 - 30
                    animation_label.setGeometry(x, y, new_width, new_height)
                    text_y_pos = y + new_height + 10
                    loading_text = QLabel("Loading...", splash)
                    loading_text.setFont(QFont("Segoe UI", 14))
                    loading_text.setAlignment(Qt.AlignCenter)
                    text_x = (splash_width - 100) // 2
                    loading_text.setGeometry(text_x, text_y_pos, 100, 30)
                    loading_text.show()
                else:
                    print("Error: Original GIF size reported as 0x0 or invalid.")
                    splash.showMessage("Loading...", Qt.AlignCenter)
                animation_label.show()
        except Exception as e:
            print(f"An unexpected error occurred while loading the GIF: {e}")
            splash.showMessage("Loading...", Qt.AlignCenter)

    splash.show()
    app.processEvents()

    loader = MatlabEngineLoader()
    loader.engine_ready.connect(lambda engine: onEngineLoaded_animated(engine, splash))
    loader.error_occurred.connect(lambda msg: onEngineError(msg, app, splash))
    loader.start()

    sys.exit(app.exec_())


def onEngineLoaded_animated(engine, splash):
    def start_transition():
        global main_window  # Ensure the main window is not garbage collected
        try:
            main_window = MainWindow(engine)
        except Exception as e:
            onEngineError(f"Failed to create MainWindow: {e}", QApplication.instance(), splash)
            return

        main_window.setWindowOpacity(0.0)
        main_window.show()

        splash_anim = QPropertyAnimation(splash, b"windowOpacity", splash)
        splash_anim.setDuration(700)
        splash_anim.setStartValue(1.0)
        splash_anim.setEndValue(0.0)
        splash_anim.setEasingCurve(QEasingCurve.InOutQuad)
        splash_anim.finished.connect(lambda: splash.close())

        fade_in = QPropertyAnimation(main_window, b"windowOpacity", main_window)
        fade_in.setDuration(700)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.InOutQuad)

        splash_anim.start(QPropertyAnimation.DeleteWhenStopped)
        fade_in.start(QPropertyAnimation.DeleteWhenStopped)

    QTimer.singleShot(0, start_transition)


def onEngineError(error_msg, app, splash):
    """Handle MATLAB engine loading error."""
    splash.hide()
    QMessageBox.critical(None, "Error", f"Failed to load MATLAB engine: {error_msg}")
    app.quit()


if __name__ == "__main__":
    main()
