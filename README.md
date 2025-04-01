# Speech Intelligibility Experiment

A PyQt5-based application for conducting speech intelligibility experiments using HSM and OLSA tests.

## Features

- Support for both HSM and OLSA tests
- MATLAB engine integration for audio processing
- CI map loading and management
- Real-time audio streaming
- User-friendly interface

## Requirements

- Python 3.x
- PyQt5
- MATLAB Engine API for Python
- MATLAB (with required toolboxes)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SpeechExp.git
cd SpeechExp
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Ensure MATLAB is installed and properly configured with the MATLAB Engine API for Python.

## Usage

1. Run the application:
```bash
python main.py
```

2. Enter subject information and select test type
3. Load CI map data if available
4. Start the experiment

## Project Structure

- `main.py`: Main application entry point
- `core/`: Core functionality modules
  - `base_window.py`: Base window class
  - `matlab_engine.py`: MATLAB engine integration
- `hsm/`: HSM test implementation
- `olsa/`: OLSA test implementation

## License

[Your chosen license]

## Contributing

[Your contribution guidelines] 