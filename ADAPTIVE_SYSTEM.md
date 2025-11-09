# Adaptive Multi-Modal Neuroimaging Analysis System

## Overview

The **Adaptive Multi-Modal Analyzer** is a revolutionary system that **automatically adapts** to whatever neuroimaging data you provide. No more rigid, format-specific tools - this system intelligently analyzes any combination of:

- **EEG** (Electroencephalography)
- **MEG** (Magnetoencephalography)
- **MRI** (Structural MRI)
- **fMRI** (Functional MRI)
- **cMRI** (Contrast-enhanced MRI)

## Key Innovation: Adaptive Analysis

### Traditional Approach (❌)
```
if (has_eeg AND has_mri):
    run_eeg_mri_analysis()
elif (has_meg AND has_fmri):
    run_meg_fmri_analysis()
# ... 100+ combinations needed!
```

### Adaptive Approach (✅)
```python
modalities = detect_loaded_data()
analysis_engine.auto_analyze(modalities)
# Automatically runs appropriate analyses!
```

## Supported Modalities

### 1. EEG (Electroencephalography)

**What it measures**: Electrical brain activity via scalp electrodes

**Formats supported**:
- EEGLAB (.set)
- FIF (.fif)
- EDF (.edf)
- BDF (.bdf)
- BrainVision (.vhdr, .vmrk, .eeg)

**Extracted features**:
- Band power (Delta, Theta, Alpha, Beta, Gamma)
- Mean/std amplitude
- Channel connectivity
- Time-domain statistics

### 2. MEG (Magnetoencephalography)

**What it measures**: Magnetic fields from brain activity

**Formats supported**:
- FIF (.fif)
- CTF (.ds)
- BTi/4D (.pdf)

**Extracted features**:
- Similar to EEG
- Higher spatial resolution
- Source-space features (if available)

### 3. MRI (Structural MRI)

**What it measures**: Brain anatomy/structure

**Formats supported**:
- NIfTI (.nii, .nii.gz)
- FreeSurfer (.mgz, .mgh)
- DICOM (.dcm) - via nibabel

**Extracted features**:
- Mean/std/variance intensity
- Skewness, kurtosis
- Regional volumes and intensities
- Voxel-based statistics

### 4. fMRI (Functional MRI)

**What it measures**: BOLD signal (blood oxygen level dependent) - brain activity

**Formats supported**:
- NIfTI (.nii, .nii.gz) - 4D files
- Preprocessed fMRI data

**Extracted features**:
- Mean BOLD activation
- Temporal variance
- Regional activation patterns
- Time series for each voxel/region

### 5. cMRI (Contrast-enhanced MRI)

**What it measures**: Enhanced anatomical details with contrast agent

**Formats supported**:
- NIfTI (.nii, .nii.gz)
- DICOM (.dcm)

**Extracted features**:
- Contrast enhancement ratio
- High-intensity region detection
- Regional contrast patterns
- Enhancement distribution

## How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│         User Interface Layer            │
│  (Adaptive buttons/tabs based on data)  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Modality Manager                   │
│  - Detects loaded data types            │
│  - Manages multiple modalities          │
│  - Tracks data status                   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    Feature Extraction Engine            │
│  - Type-specific feature extraction     │
│  - Automatic feature selection          │
│  - Normalization and scaling            │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Adaptive Analysis Engine            │
│  - Selects appropriate analyses         │
│  - Single vs multi-modal logic          │
│  - Cross-modal correlations             │
│  - Non-linear dynamics                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    Visualization Engine                 │
│  - Modality-specific plots              │
│  - Cross-modal matrices                 │
│  - Adaptive dashboards                  │
└─────────────────────────────────────────┘
```

### Data Flow

1. **Load**: User loads any combination of modalities
2. **Detect**: System identifies what's loaded
3. **Extract**: Type-specific feature extraction
4. **Analyze**: Appropriate analyses selected automatically
5. **Visualize**: Results displayed with adaptive UI

## Usage Examples

### Example 1: EEG + MRI

**Scenario**: You have EEG data and structural MRI from the same subject

**Steps**:
```bash
python adaptive_multimodal.py
```

1. Click "Load" under EEG → Select your .set file
2. Click "Load" under MRI → Select your .nii file
3. Click "Auto-Analyze"

**What happens automatically**:
- ✓ EEG features extracted (band powers, connectivity)
- ✓ MRI features extracted (regional intensities)
- ✓ Cross-modal correlation computed
- ✓ Spectral-spatial relationships analyzed
- ✓ Results visualized in correlation matrix

**Output**:
- EEG tab: Time series, power spectra
- MRI tab: 3-slice view
- Cross-Modal tab: Correlation matrix
- Results: Detailed metrics

### Example 2: MEG + fMRI

**Scenario**: MEG recordings with functional MRI

**Steps**:
1. Load MEG (.fif file)
2. Load fMRI (4D .nii file)
3. Click "Auto-Analyze"

**What happens automatically**:
- ✓ MEG spectral analysis
- ✓ fMRI BOLD activation patterns
- ✓ Temporal correlations between MEG and BOLD
- ✓ Source-space to voxel-space mapping (if possible)

**Use case**: Find which brain regions (fMRI) correspond to specific MEG oscillations

### Example 3: EEG + MEG + MRI

**Scenario**: Multi-modal epilepsy study

**Steps**:
1. Load EEG
2. Load MEG
3. Load structural MRI
4. Click "Auto-Analyze"

**What happens automatically**:
- ✓ EEG-MEG correlation (validate measurements)
- ✓ EEG-MRI correlation (electrical-structural)
- ✓ MEG-MRI correlation (magnetic-structural)
- ✓ 3×3 cross-modal matrix
- ✓ Multivariate analysis across all three

**Use case**: Comprehensive epilepsy focus localization

### Example 4: fMRI + cMRI

**Scenario**: Tumor study with functional and contrast MRI

**Steps**:
1. Load fMRI (BOLD activation)
2. Load cMRI (contrast-enhanced)
3. Click "Auto-Analyze"

**What happens automatically**:
- ✓ BOLD activation map
- ✓ Contrast enhancement regions
- ✓ Correlation between activation and enhancement
- ✓ Identify functional areas near tumor

**Use case**: Pre-surgical planning

### Example 5: All Five Modalities

**Scenario**: Comprehensive brain mapping study

**Steps**:
1. Load EEG
2. Load MEG
3. Load MRI (structural)
4. Load fMRI (functional)
5. Load cMRI (if available)
6. Click "Auto-Analyze"

**What happens automatically**:
- ✓ 5×5 cross-modal correlation matrix
- ✓ Canonical correlation analysis
- ✓ Multivariate pattern analysis
- ✓ Structure-function relationships
- ✓ Multi-level integration

## Adaptive Analysis Logic

The system automatically selects analyses based on what's loaded:

### Single Modality Loaded

**Available analyses**:
- Descriptive statistics
- Feature extraction
- Complexity measures (if electrophysiology)
- Regional analysis (if imaging)

**Example output**:
```
Single Modality Analysis: EEG
Extracted Features:
  power_alpha: mean=12.45, std=3.21
  power_beta: mean=8.76, std=2.15
  connectivity: shape=(64, 64)
```

### Two Modalities Loaded

**Available analyses**:
- Pairwise correlation
- Linear regression
- Canonical correlation
- Non-linear coupling (if enabled)
- Spectral-spatial mapping (if applicable)

**Example output**:
```
Cross-Modal Analysis: EEG ↔ MRI
Correlation: 0.67 (p < 0.001)
Alpha power correlates with frontal gray matter density
```

### Three+ Modalities Loaded

**Available analyses**:
- All pairwise correlations
- Multi-way correlation matrix
- Multivariate analysis
- Principal component analysis across modalities
- Network analysis

**Example output**:
```
Cross-Modal Correlation Matrix:
       EEG    MEG    MRI
EEG   1.00   0.82   0.65
MEG   0.82   1.00   0.71
MRI   0.65   0.71   1.00

Strongest relationship: EEG ↔ MEG (0.82)
All correlations significant (p < 0.01)
```

## Feature Extraction Details

### Electrophysiology (EEG/MEG)

```python
features = {
    'power_delta': [array of delta power per channel],
    'power_theta': [array of theta power per channel],
    'power_alpha': [array of alpha power per channel],
    'power_beta': [array of beta power per channel],
    'power_gamma': [array of gamma power per channel],
    'mean_amplitude': [array],
    'std_amplitude': [array],
    'connectivity': [channel × channel matrix]
}
```

### Structural Imaging (MRI/cMRI)

```python
features = {
    'mean_intensity': scalar,
    'std_intensity': scalar,
    'variance': scalar,
    'skewness': scalar,
    'kurtosis': scalar,
    'regional_means': [8 regions],
    'regional_stds': [8 regions],
    'regional_volumes': [8 regions]
}
```

### Functional Imaging (fMRI)

```python
features = {
    'mean_bold': scalar,
    'std_bold': scalar,
    'temporal_variance': scalar,
    'regional_activation': [8 regions],
    'temporal_signal': [time series]
}
```

## Cross-Modal Correlations

### How Correlation is Computed

For each pair of modalities:

1. **Feature Alignment**: Find comparable features
   - Both have regional data → correlate regional means
   - Both have spectral data → correlate band powers
   - Mixed types → correlate summary statistics

2. **Normalization**: Z-score normalization

3. **Correlation**: Pearson or Spearman

4. **Significance**: P-value computation

### Example: EEG-MRI Correlation

```python
# EEG regional features (8 channels/regions)
eeg_regional = [12.5, 14.2, 11.8, 13.9, 15.1, 12.3, 14.7, 13.2]

# MRI regional intensities (8 regions)
mri_regional = [245, 267, 238, 259, 278, 241, 265, 253]

# Correlation
r, p = pearsonr(eeg_regional, mri_regional)
# r = 0.94, p = 0.0006
```

**Interpretation**: Strong positive correlation - regions with higher MRI intensity show higher EEG alpha power

## Visualization Outputs

### Modality-Specific Tabs

Each loaded modality gets its own tab with appropriate visualization:

**EEG/MEG**:
- Time series plot (scrollable)
- Power spectral density
- Topographic maps (if electrode locations available)

**MRI/cMRI**:
- Three orthogonal slices (sagittal, coronal, axial)
- Intensity histogram
- 3D rendering (optional)

**fMRI**:
- Mean BOLD activation maps
- Time series plot
- Activation overlay on anatomy

### Cross-Modal Visualization

**Correlation Matrix**:
- N×N heatmap (N = number of loaded modalities)
- Color-coded by correlation strength
- Annotated with correlation values
- Hierarchical clustering (optional)

**Scatter Plots**:
- For each modality pair
- Regional data as points
- Regression line
- 95% confidence interval

## Advanced Features

### Canonical Correlation Analysis (CCA)

When 2+ modalities are loaded, CCA finds linear combinations that maximize correlation:

```python
# Automatically applied when appropriate
cca = CCA(n_components=2)
cca.fit(features_modality1, features_modality2)

# Find canonical variates
canonical_corr = cca.score(test_data1, test_data2)
```

**Use**: Find hidden relationships between modality patterns

### Principal Component Analysis

Reduces dimensionality across all modalities:

```python
# Combined feature matrix
all_features = concatenate([eeg_features, mri_features, fmri_features])

pca = PCA(n_components=0.95)  # Keep 95% variance
components = pca.fit_transform(all_features)
```

**Use**: Identify main patterns of variation

### Non-Linear Analysis Integration

If `nonlinear_analysis.py` is available:

- Mutual information between modalities
- Phase synchronization (for electrophysiology)
- Transfer entropy (directionality)
- Complexity measures for each modality

## Advantages Over Traditional Tools

| Feature | Traditional Tools | Adaptive System |
|---------|-------------------|-----------------|
| **Flexibility** | Fixed format | Any combination |
| **Code** | Separate scripts per combo | Single unified tool |
| **Maintenance** | Update N scripts | Update once |
| **Learning curve** | Learn N tools | Learn one interface |
| **Analysis** | Manual selection | Automatic |
| **Extensibility** | Add new script | Add new modality class |

## Technical Implementation

### Modular Architecture

```python
class ModalityData:
    """Self-contained modality with its own feature extraction"""

    def extract_features(self):
        if self.type == "EEG":
            return self._extract_electrophysiology()
        elif self.type == "MRI":
            return self._extract_structural()
        # ... automatically dispatches to right method
```

### Adaptive UI

```python
# UI adapts to loaded data
for modality_type in [EEG, MEG, MRI, fMRI, cMRI]:
    if modality_type in loaded_modalities:
        create_tab(modality_type)
        enable_analysis_buttons()
```

### Feature Compatibility Matrix

```python
compatibility = {
    ('EEG', 'MEG'): ['band_powers', 'connectivity'],
    ('EEG', 'MRI'): ['regional_means'],
    ('fMRI', 'MRI'): ['regional_activation_vs_structure'],
    # ... defines what can be compared
}
```

## File Structure

```
mne-workout/
├── adaptive_multimodal.py          # Main adaptive system ⭐
├── multimodal_analysis.py          # Original multi-modal tool
├── nonlinear_analysis.py           # Non-linear methods
├── ADAPTIVE_SYSTEM.md              # This guide
├── NONLINEAR_DYNAMICS.md           # Non-linear documentation
├── MULTIMODAL_ANALYSIS.md          # Multi-modal guide
└── README.md                        # Main documentation
```

## Common Use Cases

### Research

1. **Multi-modal brain mapping**: Load all available modalities
2. **Method validation**: Compare EEG vs MEG
3. **Structure-function**: Correlate anatomy (MRI) with activity (fMRI/EEG)

### Clinical

1. **Epilepsy**: EEG + MRI for seizure focus localization
2. **Tumors**: cMRI + fMRI for surgical planning
3. **Stroke**: MRI + fMRI for functional recovery
4. **Dementia**: Structural (MRI) + functional (EEG/MEG) decline

### Quality Control

1. **Data validation**: Load multiple acquisitions, check consistency
2. **Artifact detection**: Compare modalities for anomalies
3. **Registration check**: Verify alignment across modalities

## Troubleshooting

### "Correlation is zero/very low"

**Causes**:
- Different subjects (data not from same person)
- Different time points
- Poor data quality
- No true relationship

**Solution**: Verify data correspondence

### "Cannot extract features"

**Causes**:
- Unsupported file format variant
- Corrupted data
- Missing metadata

**Solution**: Check file integrity, try re-exporting

### "Analysis takes too long"

**Causes**:
- Very large datasets (esp. fMRI)
- Many modalities loaded

**Solution**:
- Downsample data
- Close other applications
- Process regions of interest only

## Future Enhancements

Planned features:
- DICOM support for all modalities
- Diffusion MRI (DTI, DWI)
- PET/SPECT integration
- Real-time data streaming
- Cloud-based analysis
- GPU acceleration
- Machine learning predictions

## References

**Multi-modal integration**:
- Calhoun et al. (2009). "Multimodal fusion of brain imaging data"
- Sui et al. (2012). "A review of multivariate methods for multimodal fusion"

**Canonical correlation**:
- Correa et al. (2010). "Canonical correlation analysis for data fusion"

## Citation

If you use this tool in research:
```
Adaptive Multi-Modal Neuroimaging Analyzer
Supports EEG, MEG, MRI, fMRI, and contrast MRI
Automatic analysis selection based on available data
```

## Quick Start

```bash
# Run the adaptive system
python adaptive_multimodal.py

# Load your data (any combination):
# 1. Click "Load" buttons for your modalities
# 2. Select files from disk
# 3. Click "Auto-Analyze"
# 4. View results in generated tabs

# That's it! The system handles everything else.
```

## Summary

The Adaptive Multi-Modal Analyzer represents a **paradigm shift** in neuroimaging analysis:

✅ **One tool** instead of many format-specific scripts
✅ **Automatic** analysis selection
✅ **Any combination** of modalities
✅ **Intelligent** feature extraction
✅ **Robust** to missing data
✅ **Extensible** architecture
✅ **User-friendly** interface

**Load your data → Click Auto-Analyze → Get results!**
