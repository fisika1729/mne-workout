# MNE-Python Data Analyzer

A Python application for analyzing MRI/EEG data using the MNE-Python library. This tool provides an interactive GUI for loading, visualizing, and analyzing EEGLAB .set format files.

## Features

- **File Browser**: Easy-to-use tkinter-based file selection dialog
- **Multiple Visualizations**:
  - Raw data plotting with customizable time windows
  - Power Spectral Density (PSD) analysis
  - Sensor location visualization
  - Detailed data information display
- **Interactive GUI**: Built with tkinter and matplotlib for cross-platform compatibility
- **Path Management**: Uses `os` and `pathlib` for robust file handling

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd mne-workout
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install mne numpy scipy matplotlib pandas scikit-learn seaborn
```

## Usage

### Running the Application

Simply run the main script:
```bash
python mri_analysis.py
```

Or make it executable and run:
```bash
chmod +x mri_analysis.py
./mri_analysis.py
```

### Loading Data

1. Click the **"Load .set File"** button or use the File menu
2. Navigate to your .set file location using the file browser
3. Select your EEGLAB .set format file
4. The data will be automatically loaded and displayed

### Analyzing Data

The application provides three tabs:

#### 1. Raw Data Tab
- View raw EEG/MEG signals
- Adjust the time window using "Duration" and "Start" parameters
- Click "Plot Raw Data" to refresh the visualization
- Use "Plot Sensors" to view electrode/sensor locations

#### 2. Power Spectral Density Tab
- Analyze frequency content of your signals
- Automatically computed when data is loaded

#### 3. Data Info Tab
- View detailed information about your dataset:
  - Number of channels
  - Sampling rate
  - Duration
  - Channel names and types

## File Format Support

Currently supports:
- **EEGLAB .set files**: The primary format for this tool

The tool uses MNE-Python's `read_raw_eeglab()` function, which expects:
- `.set` file (header file)
- `.fdt` file (data file, if separate)

## Dependencies

- **mne**: Core library for neurophysiological data analysis
- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **matplotlib**: Plotting and visualization
- **tkinter**: GUI framework (usually included with Python)
- **pathlib & os**: File path management

## Example Data

If you don't have .set files, you can:
1. Download sample data from [EEGLAB website](https://sccn.ucsd.edu/eeglab/)
2. Use MNE's built-in sample datasets:
```python
import mne
sample_data_path = mne.datasets.sample.data_path()
```

## Troubleshooting

### Common Issues

**Issue**: "Failed to load file" error
- **Solution**: Ensure both .set and .fdt files are in the same directory

**Issue**: "Could not plot sensor locations"
- **Solution**: Your data may not include electrode coordinates. This is normal for some datasets.

**Issue**: tkinter not found
- **Solution**:
  - Ubuntu/Debian: `sudo apt-get install python3-tk`
  - MacOS: tkinter comes with Python
  - Windows: tkinter comes with Python

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## References

- [MNE-Python Documentation](https://mne.tools/)
- [EEGLAB](https://sccn.ucsd.edu/eeglab/)
- [Matplotlib Documentation](https://matplotlib.org/)