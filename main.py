import os
import sys
import traceback
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QFont, QPixmap, QMovie, QColor, QPainter, QLinearGradient
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLineEdit, QLabel, QComboBox, QGridLayout, QVBoxLayout, QHBoxLayout,
    QSpacerItem, QSizePolicy, QMessageBox, QSplashScreen, QFileDialog, QProgressBar, QPushButton, QGraphicsDropShadowEffect
)

# Import your existing modules (these are assumed to exist as in your original code)
from core.matlab_engine import MatlabEngineLoader
from core.base_window import BaseWindow
from hsm.hsm_window import HSMTestWindow
from olsa.olsa_window import OLSATestWindow


from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QPushButton, QGraphicsDropShadowEffect

class CoolPushButton(QPushButton):
    """A QPushButton subclass that adds a smoothly animated drop-shadow effect on hover."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        # Create and attach the shadow effect with an initial blur radius of 0.
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(0)
        self._shadow.setOffset(0, 0)
        self._shadow.setColor(QColor(255, 255, 255, 80))
        self.setGraphicsEffect(self._shadow)
        
        # Setup the property animation for the blurRadius property.
        self._animation = QPropertyAnimation(self._shadow, b"blurRadius", self)
        self._animation.setDuration(150)  # Duration in milliseconds; adjust as desired.
        self._animation.setEasingCurve(QEasingCurve.InOutCubic)
    
    def enterEvent(self, event):
        # Animate to a blur radius of 20 when the mouse enters.
        self._animation.stop()
        self._animation.setStartValue(self._shadow.blurRadius())
        self._animation.setEndValue(20)
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
        self.initUI()
        self.hide()  # Do not show immediately
        self.setObjectName("mainWindow")
        self.setFixedSize(1300, 750)
        self.center()
        self.applyStyles()

    def initUI(self):
        """Initialize and build the GUI elements."""
        super().initUI()

        # Title label
        titleLabel = QLabel("Speech Intelligibility Experiment")
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

        nameLabel = QLabel("Name:")
        nameLabel.setStyleSheet("color: #ececec;")
        self.nameInput = QLineEdit("wolf-dieter")

        surnameLabel = QLabel("Surname:")
        surnameLabel.setStyleSheet("color: #ececec;")
        self.surnameInput = QLineEdit("Goecke")

        idLabel = QLabel("ID:")
        idLabel.setStyleSheet("color: #ececec;")
        self.idInput = QLineEdit("12345")

        testLabel = QLabel("Select Test:")
        testLabel.setStyleSheet("color: #ececec;")
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

        self.mapDataLayout = QGridLayout()
        self.mapDataLayout.setVerticalSpacing(15)
        self.mapDataLayout.setHorizontalSpacing(20)

        thresholdLabel = QLabel("Thresholds (T Levels):")
        thresholdLabel.setStyleSheet("color: #ececec;")
        thresholdLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        thresholdLabel.setFixedWidth(220)

        comfortableLabel = QLabel("Comfortable Levels (C Levels):")
        comfortableLabel.setStyleSheet("color: #ececec;")
        comfortableLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        comfortableLabel.setFixedWidth(220)

        self.lowerLevelsLabel = QLabel("Not loaded")
        self.lowerLevelsLabel.setStyleSheet(
            "background-color: #2a2a2a; padding: 6px 10px; border-radius: 3px; color: #ececec;"
        )
        self.lowerLevelsLabel.setMinimumHeight(36)
        self.lowerLevelsLabel.setMinimumWidth(600)
        self.lowerLevelsLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.upperLevelsLabel = QLabel("Not loaded")
        self.upperLevelsLabel.setStyleSheet(
            "background-color: #2a2a2a; padding: 6px 10px; border-radius: 3px; color: #ececec;"
        )
        self.upperLevelsLabel.setMinimumHeight(36)
        self.upperLevelsLabel.setMinimumWidth(600)
        self.upperLevelsLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        stimRateLabel = QLabel("Channel Stimulation Rate (Hz):")
        stimRateLabel.setStyleSheet("color: #ececec;")
        stimRateLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        stimRateLabel.setFixedWidth(220)

        self.channelStimRateLabel = QLabel("Not loaded")
        self.channelStimRateLabel.setStyleSheet(
            "background-color: #2a2a2a; padding: 6px 10px; border-radius: 3px; color: #ececec;"
        )
        self.channelStimRateLabel.setMinimumHeight(36)
        self.channelStimRateLabel.setMinimumWidth(600)
        self.channelStimRateLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.mapDataLayout.addWidget(thresholdLabel, 0, 0)
        self.mapDataLayout.addWidget(self.lowerLevelsLabel, 0, 1, 1, 3)
        self.mapDataLayout.addWidget(comfortableLabel, 1, 0)
        self.mapDataLayout.addWidget(self.upperLevelsLabel, 1, 1, 1, 3)
        self.mapDataLayout.addWidget(stimRateLabel, 2, 0)
        self.mapDataLayout.addWidget(self.channelStimRateLabel, 2, 1, 1, 3)

        mapDataContainer.addLayout(self.mapDataLayout)
        mapAndCalibrationContainer.addLayout(mapDataContainer, 4)  # Ratio 4:1 for Map Data

        # RIGHT SIDE: Calibration Section
        self.calibrationWidget = QWidget()
        self.calibrationWidget.setStyleSheet(
            "background-color: #333333; border: 2px solid #ff9900; border-radius: 8px; padding: 8px;"
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
            "color: #ffffff; background-color: #1e1e1e; padding: 6px; border-radius: 3px;"
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
        for i in range(self.mapDataLayout.count()):
            item = self.mapDataLayout.itemAt(i)
            if item.widget():
                item.widget().setVisible(False)
        self.calibrationWidget.setVisible(False)

        self.layout.addSpacing(20)
        self.layout.addSpacerItem(
            QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Action Buttons
        btnLayout = QGridLayout()
        btnLayout.setSpacing(20)

        self.startButton = CoolPushButton("Start Experiment")
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

        self.startButton.setEnabled(True)
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
            QMessageBox.warning(
                self,
                "Error",
                "MATLAB engine is not ready. Please wait for initialization.",
            )
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
            self.loadMapButton.setEnabled(False)

            self.map_data = self.eng.call_map(name, surname, side, mapNumber, nargout=1)
            lower_levels = self.map_data["lower_levels"]
            upper_levels = self.map_data["upper_levels"]
            channel_stim_rate = self.map_data["channel_stim_rate"]

            # Store original upper levels and prepare numeric versions
            self.original_upper_levels = upper_levels
            self.original_upper_levels_numeric = self.extract_values_from_matlab(upper_levels)
            self.calibrated_upper_levels_numeric = self.original_upper_levels_numeric[:]

            self.lowerLevelsLabel.setText(self._clean_matlab_array(lower_levels))
            self.upperLevelsLabel.setText(self._clean_matlab_array(upper_levels))
            self.channelStimRateLabel.setText(self._clean_matlab_array(channel_stim_rate))

            self.mapDataGroupLabel.setVisible(True)
            for i in range(self.mapDataLayout.count()):
                item = self.mapDataLayout.itemAt(i)
                if item.widget():
                    item.widget().setVisible(True)

            self.setFixedSize(1300, 750)
            self.center()

            self.startButton.setText("Start Experiment (Map Loaded)")
            self.startButton.setStyleSheet("background-color: #28a745;")
            self.mapDataGroupLabel.setText("CI Map Data")
            self.mapDataGroupLabel.setStyleSheet("color: #28a745;")
            QApplication.processEvents()

            self.showCalibrationControls()
            self.updateCalibration()
        except Exception as e:
            QMessageBox.critical(self, "MATLAB Error", str(e))
            self.map_data = None
        finally:
            self.loadMapButton.setEnabled(True)
            self.statusLabel.setText("Ready")

    def startTest(self):
        """Start the selected test."""
        name = self.nameInput.text()
        surname = self.surnameInput.text()
        id_val = self.idInput.text()
        test_type = self.testCombo.currentText()
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
            msg.setStyleSheet(
                """
                QMessageBox { background-color: #2c3e50; }
                QMessageBox QLabel { color: white; font-size: 12px; padding: 10px; }
                QPushButton { background-color: #3498db; color: white; border: none;
                             padding: 5px 15px; border-radius: 3px; min-width: 80px; }
                QPushButton:hover { background-color: #2980b9; }
                """
            )
            msg.exec_()
            return

        if test_type == "HSM":
            self.test_window = HSMTestWindow(
                self.eng, name, surname, id_val, self.map_data, map_number, map_side
            )
        else:
            self.test_window = OLSATestWindow(
                self.eng, name, surname, id_val, self.map_data, map_number, map_side
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
        QMessageBox.information(self, "Help", help_text)

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
                background-color: #141414;
            }
            QLabel {
                color: #ececec;
            }
            QPushButton {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #ff6a00, stop:1 #ee0979);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #ff7f24, stop:1 #ff1493);
            }
            QPushButton:pressed {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #ff4500, stop:1 #c71585);
            }
            QComboBox {
                background-color: #282828;
                color: #ececec;
                border: 1px solid #444;
                padding: 5px;
                border-radius: 8px;
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
                border: 1px solid #444;
                border-radius: 8px;
                text-align: center;
                background-color: #282828;
            }
            QProgressBar::chunk {
                background-color: #ff6a00;
                border-radius: 8px;
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
            QMessageBox.warning(self, "Error", "No map data loaded. Please load a map first.")
            return

        try:
            QMessageBox.information(
                self,
                "Test Sentence",
                "Playing test sentence with current calibration settings.",
            )
            # Insert actual call to MATLAB function here
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error playing test sentence: {e}")

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
        """Update the calibrated levels based on the calibration percentage."""
        if not self.original_upper_levels_numeric:
            print("No original upper levels data for calibration")
            return

        try:
            self.calibrationPercentLabel.setText(f"{self.calibration_percent}%")
            calibrated_values = [
                val * (self.calibration_percent / 100.0)
                for val in self.original_upper_levels_numeric
            ]
            self.calibrated_upper_levels_numeric = calibrated_values
            formatted_values = [f"{val:.1f}" for val in self.calibrated_upper_levels_numeric]
            self.upperLevelsLabel.setText(", ".join(formatted_values))
        except Exception as e:
            print(f"Error in updateCalibration: {e}")
            traceback.print_exc()


def main():
    app = QApplication(sys.argv)

    # Create a splash pixmap with a modern gradient background
    splash_width, splash_height = 640, 480
    splash_pix = QPixmap(splash_width, splash_height)
    splash_pix.fill(Qt.transparent)
    painter = QPainter(splash_pix)
    gradient = QLinearGradient(0, 0, splash_width, splash_height)
    gradient.setColorAt(0, QColor("#1f1c2c"))
    gradient.setColorAt(1, QColor("#928dab"))
    painter.fillRect(0, 0, splash_width, splash_height, gradient)
    painter.end()

    splash = CustomSplashScreen(splash_pix)
    splash.setWindowFlags(splash.windowFlags() | Qt.WindowStaysOnTopHint)
    splash.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")

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
    gif_path = os.path.join(script_dir, "assets", "loading.gif")
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
