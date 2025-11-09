# MNE-Python Data Analyzer

A comprehensive Python application suite for analyzing MRI/EEG data using the MNE-Python library. Supports **multi-modal analysis** with correlation between structural (MRI) and functional (EEG) brain data.

## Tools Included

### 1. Basic EEG Analyzer (`mri_analysis.py`)
- Single-modality EEG/MEG analysis
- EEGLAB .set format support
- Quick visualization and basic analysis

### 2. Multi-Modal Analyzer (`multimodal_analysis.py`) ⭐ **NEW**
- **Supports both MRI and EEG data**
- **Multiple file formats** (EEGLAB, FIF, EDF, NIfTI, etc.)
- **Correlation analysis** between MRI structure and EEG function
- **Advanced visualizations** with statistical significance testing

### 3. Command-Line Tool (`example_usage.py`)
- Batch processing
- Automated report generation
- Scriptable analysis

## Features

### Basic Features
- **File Browser**: Easy-to-use tkinter-based file selection dialog
- **Multiple Visualizations**:
  - Raw data plotting with customizable time windows
  - Power Spectral Density (PSD) analysis
  - Sensor location visualization
  - Detailed data information display
- **Interactive GUI**: Built with tkinter and matplotlib for cross-platform compatibility
- **Path Management**: Uses `os` and `pathlib` for robust file handling

### Multi-Modal Features ⭐
- **Multi-Format Support**:
  - **EEG**: .set (EEGLAB), .fif (MNE), .edf, .bdf, .vhdr (BrainVision)
  - **MRI**: .nii/.nii.gz (NIfTI), .mgz/.mgh (FreeSurfer)
- **Correlation Analysis**:
  - Spectral-spatial correlations (EEG frequency bands vs MRI regions)
  - Statistical significance testing (p-values)
  - Feature extraction from both modalities
- **Advanced Visualizations**:
  - EEG band power analysis (Delta, Theta, Alpha, Beta, Gamma)
  - MRI 3D slice viewing (Sagittal, Coronal, Axial)
  - Correlation heatmaps and scatter plots
  - Channel connectivity matrices
- **Find Similarities**: Automatically identifies correlations between brain structure and function

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

### Running the Basic EEG Analyzer

Simply run the main script:
```bash
python mri_analysis.py
```

Or make it executable and run:
```bash
chmod +x mri_analysis.py
./mri_analysis.py
```

### Running the Multi-Modal Analyzer ⭐ **RECOMMENDED**

For MRI/EEG correlation analysis:
```bash
python multimodal_analysis.py
```

**Workflow:**
1. Click "Load EEG" → Select your EEG file (any supported format)
2. Click "Load MRI" → Select your MRI file (.nii, .mgz, etc.)
3. Click "Correlate" → Computes correlations between MRI and EEG
4. View results in "Correlation Analysis" tab

**What it does:**
- Extracts EEG frequency band power (Delta, Theta, Alpha, Beta, Gamma)
- Divides MRI into regions and computes intensity statistics
- Correlates EEG spectral features with MRI spatial features
- Displays statistical significance (p-values)
- Shows which brain regions correlate with specific EEG frequencies

### Running the Command-Line Tool

For batch processing:
```bash
python example_usage.py /path/to/your/data.set
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

### EEG/MEG Formats
- **EEGLAB** (.set) - Header and data files
- **FIF** (.fif) - MNE-Python native format
- **EDF** (.edf) - European Data Format
- **BDF** (.bdf) - BioSemi Data Format
- **BrainVision** (.vhdr, .vmrk, .eeg) - Brain Products

### MRI Formats
- **NIfTI** (.nii, .nii.gz) - Standard neuroimaging format
- **FreeSurfer** (.mgz, .mgh) - FreeSurfer MRI format
- Supports T1, T2, FLAIR, and other structural sequences

### Notes
- For EEGLAB: Both `.set` (header) and `.fdt` (data) files should be in same directory
- For BrainVision: All three files (.vhdr, .vmrk, .eeg) must be present
- MRI files are loaded using nibabel library

## Dependencies

- **mne**: Core library for neurophysiological data analysis
- **numpy**: Numerical computations
- **scipy**: Scientific computing and statistics
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning (PCA, feature scaling)
- **nibabel**: MRI file I/O (NIfTI, FreeSurfer formats)
- **tkinter**: GUI framework (usually included with Python)
- **pathlib & os**: File path management

See `requirements.txt` for complete list with versions.

## Example Data

If you don't have .set files, you can:
1. Download sample data from [EEGLAB website](https://sccn.ucsd.edu/eeglab/)
2. Use MNE's built-in sample datasets:
```python
import mne
sample_data_path = mne.datasets.sample.data_path()
```

## Understanding Multi-Modal Correlations

### What the Tool Computes

The multi-modal analyzer finds correlations between:
1. **EEG Frequency Band Power** (Delta, Theta, Alpha, Beta, Gamma)
2. **MRI Regional Intensities** (brain regions divided spatially)

### Interpreting Results

**Correlation Coefficient**:
- `+0.7 to +1.0`: Strong positive correlation
- `+0.3 to +0.7`: Moderate positive correlation
- `-0.3 to +0.3`: Weak or no correlation
- `-0.7 to -0.3`: Moderate negative correlation
- `-1.0 to -0.7`: Strong negative correlation

**P-value**:
- `< 0.05`: Statistically significant (marked in green)
- `≥ 0.05`: Not statistically significant (may be due to chance)

### Example Interpretation

**Finding**: Alpha band power correlates with regional MRI intensity (r=0.65, p=0.01)

**Interpretation**:
- Regions with higher MRI intensity show stronger alpha oscillations
- This is statistically significant (p < 0.05)
- May indicate structural basis for alpha rhythm generation

### Applications

1. **Clinical**: Identify abnormal structure-function relationships
2. **Research**: Understand neural correlates of cognition
3. **Quality Control**: Verify data integrity across modalities

For detailed analysis guide, see [`MULTIMODAL_ANALYSIS.md`](MULTIMODAL_ANALYSIS.md)

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

**Issue**: "Failed to load MRI file" or nibabel not found
- **Solution**: Install nibabel: `pip install nibabel`

**Issue**: Correlations are very weak or not significant
- **Causes**:
  - EEG and MRI from different subjects
  - Poor data quality
  - Mismatched data types
  - No true structural-functional relationship
- **Solution**: Verify both datasets are from the same subject at similar time points

**Issue**: Memory error with large MRI files
- **Solution**:
  - Downsample MRI before analysis
  - Use smaller ROIs (regions of interest)
  - Close other applications to free memory

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## References

- [MNE-Python Documentation](https://mne.tools/)
- [EEGLAB](https://sccn.ucsd.edu/eeglab/)
- [Matplotlib Documentation](https://matplotlib.org/)