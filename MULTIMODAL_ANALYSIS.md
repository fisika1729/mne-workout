# Multi-Modal MRI/EEG Analysis Guide

## Overview

This advanced tool analyzes **both MRI and EEG data** to find correlations and similarities between structural (MRI) and functional (EEG) brain data. It uses `os` and `pathlib` for robust file handling across multiple data formats.

## Supported File Formats

### EEG Formats
- **EEGLAB** (.set) - Most common EEG format
- **FIF** (.fif) - MNE-Python native format
- **EDF** (.edf) - European Data Format
- **BDF** (.bdf) - BioSemi Data Format
- **BrainVision** (.vhdr) - Brain Products format

### MRI Formats
- **NIfTI** (.nii, .nii.gz) - Neuroimaging Informatics Technology Initiative
- **FreeSurfer** (.mgz, .mgh) - FreeSurfer MRI format
- Support for structural MRI (T1, T2, etc.)

## Features

### 1. Multi-Format Data Loading
- Automatic format detection based on file extension
- Uses `pathlib.Path` for cross-platform compatibility
- File browser with format filtering

### 2. EEG Analysis
- **Time-domain visualization**: Raw signal plotting
- **Frequency analysis**: Power spectral density (PSD)
- **Band power extraction**: Delta, Theta, Alpha, Beta, Gamma
- **Connectivity analysis**: Inter-channel correlations

### 3. MRI Analysis
- **3D visualization**: Sagittal, coronal, and axial slices
- **Regional analysis**: Automatic brain region division
- **Statistical features**: Mean, std, skewness, kurtosis
- **Spatial distribution**: Voxel intensity analysis

### 4. Correlation Analysis
The tool computes several types of correlations:

#### Spectral-Spatial Correlation
Correlates EEG frequency band power with MRI regional intensities:
- Delta band (0.5-4 Hz) vs MRI regions
- Theta band (4-8 Hz) vs MRI regions
- Alpha band (8-13 Hz) vs MRI regions
- Beta band (13-30 Hz) vs MRI regions
- Gamma band (30-50 Hz) vs MRI regions

#### Statistical Significance
- Pearson correlation coefficients
- P-values for significance testing
- P < 0.05 indicates significant correlation

#### Feature Extraction
**EEG Features:**
- Power in each frequency band per channel
- Mean amplitude
- Standard deviation
- Channel connectivity matrix

**MRI Features:**
- Mean intensity (overall and regional)
- Intensity variance
- Skewness and kurtosis
- Regional statistical properties

### 5. Visualization Dashboard
Six comprehensive plots:

1. **EEG Band Powers**: Bar chart of average power in each frequency band
2. **MRI Regional Intensities**: Intensity distribution across brain regions
3. **Correlation Coefficients**: Spectral-spatial correlations with significance markers
4. **EEG Connectivity Matrix**: Heatmap of inter-channel correlations
5. **Feature Scatter Plot**: Direct comparison (e.g., Alpha power vs MRI intensity)
6. **Summary Statistics**: Key findings and interpretation

## Usage Workflow

### Step 1: Launch Application
```bash
python multimodal_analysis.py
```

### Step 2: Load EEG Data
1. Click "Load EEG" button
2. Navigate to your EEG file (any supported format)
3. View automatic visualization and statistics

### Step 3: Load MRI Data
1. Click "Load MRI" button
2. Navigate to your MRI file (.nii, .mgz, etc.)
3. View 3D slices

### Step 4: Compute Correlations
1. Click "Correlate" button
2. Wait for analysis to complete
3. View results in the "Correlation Analysis" tab

### Step 5: Interpret Results
- Check correlation coefficients (-1 to +1)
- Look for green bars (statistically significant, p < 0.05)
- Read interpretation in the "Data Info" tab

## Understanding Correlations

### What Do Correlations Mean?

**Positive Correlation (+0.5 to +1.0)**
- As EEG band power increases, MRI intensity increases
- Suggests functional-structural relationship
- Example: High alpha power in regions with high gray matter density

**Negative Correlation (-0.5 to -1.0)**
- As EEG band power increases, MRI intensity decreases
- May indicate inhibitory or compensatory mechanisms

**No Correlation (close to 0)**
- No linear relationship between measures
- Could indicate independence or non-linear relationship

### Statistical Significance

- **P-value < 0.05**: Correlation is statistically significant (likely not due to chance)
- **P-value ≥ 0.05**: Correlation may be due to random variation

## Example Analysis Scenarios

### Scenario 1: Resting State Analysis
**Goal**: Find structural correlates of resting EEG
1. Load resting-state EEG data
2. Load structural T1 MRI
3. Correlate alpha band power with cortical thickness/intensity
4. High correlation might indicate structural basis for alpha rhythm

### Scenario 2: Clinical Assessment
**Goal**: Identify abnormalities in structure-function relationship
1. Load patient EEG and MRI
2. Compare correlations to normative data
3. Unusual correlations may indicate pathology

### Scenario 3: Cognitive Studies
**Goal**: Relate brain structure to EEG during tasks
1. Load task-related EEG (e.g., working memory)
2. Load structural MRI
3. Find regions where structure predicts EEG activity

## Technical Details

### Feature Extraction Methods

**EEG Band Power Calculation:**
```python
# Uses Welch's method for robust PSD estimation
freqs, psd = signal.welch(data, fs=sampling_freq, nperseg=window_length)
# Integrate power in band
band_power = mean(psd[fmin:fmax])
```

**MRI Regional Division:**
```python
# Divides brain into 8 octants
# Can be extended to atlas-based regions
regions = divide_into_octants(mri_data)
regional_stats = compute_stats_per_region(regions)
```

**Correlation Computation:**
```python
# Pearson correlation between matched features
correlation, p_value = pearsonr(eeg_features, mri_features)
```

## Limitations and Considerations

1. **Spatial Resolution**: EEG has low spatial resolution compared to MRI
   - Correlations are approximate
   - Use source localization for better spatial matching

2. **Temporal Resolution**: MRI is static, EEG is dynamic
   - MRI captures structure at one time point
   - EEG should ideally be from same time period

3. **Registration**: Data should be from same subject
   - No automatic subject matching
   - User must ensure data correspondence

4. **Statistical Power**: Requires sufficient data
   - More channels/regions = more robust correlations
   - Multiple subjects needed for group analysis

## Advanced Usage

### Custom Regional Analysis
Modify the `divide_mri_regions()` function to use anatomical atlases:
- AAL (Automated Anatomical Labeling)
- Talairach coordinates
- FreeSurfer parcellation

### Source Localization
For better spatial correspondence:
1. Use MNE's forward modeling with MRI
2. Compute inverse solution for EEG
3. Correlate source-level EEG with MRI regions

### Group-Level Analysis
Export correlation results and analyze across subjects:
```python
# Export correlation data
correlations = app.correlation_results
# Perform group statistics externally
```

## File Path Management

The application uses modern Python path handling:

```python
from pathlib import Path
import os

# Cross-platform path handling
file_path = Path(user_input)
if file_path.exists():
    parent_dir = file_path.parent
    filename = file_path.name
    extension = file_path.suffix
```

## Troubleshooting

### Issue: "Failed to load MRI file"
**Solution**: Install nibabel: `pip install nibabel`

### Issue: Low or non-significant correlations
**Possible causes:**
- Different subjects (MRI and EEG from different people)
- Different time points
- Insufficient data quality
- No true structure-function relationship

### Issue: Memory errors with large MRI files
**Solution**:
- Sample MRI data (reduce resolution)
- Use smaller regions for analysis
- Process in chunks

## References and Further Reading

1. **MNE-Python Documentation**: https://mne.tools/
2. **NiBabel (MRI I/O)**: https://nipy.org/nibabel/
3. **Structure-Function Coupling**:
   - Honey et al. (2007). "Network structure of cerebral cortex shapes functional connectivity"
   - Stam (2010). "Use of magnetoencephalography in Alzheimer's disease"

## Citation

If you use this tool in research, please cite:
- MNE-Python: Gramfort et al. (2013, 2014)
- Relevant methods papers for your analysis

## Support and Contribution

- Report issues on GitHub
- Contribute improvements via pull requests
- Share your analysis workflows with the community
