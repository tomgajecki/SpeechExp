import matlab.engine
import os
from PyQt5.QtCore import QThread, pyqtSignal

class MatlabEngineLoader(QThread):
    """Thread for loading MATLAB engine."""
    
    engine_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def run(self):
        try:
            # Start MATLAB engine
            eng = matlab.engine.start_matlab()
            
            # Add MATLAB functions directory to path
            matlab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'matlab')
            eng.addpath(matlab_path, nargout=0)
            
            # Verify MATLAB functions are available and start the server
            try:
                eng.manageStreamServer('start', nargout=0)
                self.engine_ready.emit(eng)
            except Exception as e:
                self.error_occurred.emit(f"Error starting stream server: {str(e)}")
                eng.quit()
        except Exception as e:
            self.error_occurred.emit(f"Error starting MATLAB engine: {str(e)}") 