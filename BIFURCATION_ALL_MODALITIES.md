# Bifurcation Analysis - All Five Modalities

## Quick Reference

| Modality | Bifurcation Type | What It Detects | Key Metrics |
|----------|------------------|-----------------|-------------|
| **EEG** | Temporal | State transitions, seizure onset | n_bifurcations, bifurcation_points, strengths |
| **MEG** | Temporal + Spatial Pattern | State transitions + field topology changes | global_bifurcations, spatial_pattern_bifurcations |
| **MRI** | Spatial | Tissue boundaries, lesion edges | bifurcation_map, bifurcation_density |
| **fMRI** | Spatio-Temporal | BOLD transitions + activation boundaries | temporal_bifurcations, spatial_bifurcations, spatiotemporal_bifurcations |
| **cMRI** | Enhancement + Perfusion | Tumor boundaries, perfusion transitions | enhancement_bifurcations, perfusion_bifurcations, vascular_bifurcations |

## Usage Examples

### Automatic Detection (Recommended)

```python
from nonlinear_analysis import detect_bifurcations_by_modality

# EEG
eeg_bif = detect_bifurcations_by_modality(eeg_data, 'EEG')

# MEG
meg_bif = detect_bifurcations_by_modality(meg_data, 'MEG', sfreq=1000)

# MRI
mri_bif = detect_bifurcations_by_modality(mri_data, 'MRI', method='gradient')

# fMRI
fmri_bif = detect_bifurcations_by_modality(fmri_data, 'fMRI', tr=2.0, method='spatio-temporal')

# cMRI
cmri_bif = detect_bifurcations_by_modality(cmri_data, 'cMRI', method='all')
```

### Cross-Modal Comparison (5x5 Matrix)

```python
from nonlinear_analysis import compare_bifurcations_cross_modal

# Collect results from all modalities
bifurcation_results = {
    'EEG': eeg_bif,
    'MEG': meg_bif,
    'MRI': mri_bif,
    'fMRI': fmri_bif,
    'cMRI': cmri_bif
}

# Compare
comparison = compare_bifurcations_cross_modal(bifurcation_results)

print(f"Dominant modality: {comparison['dominant_modality']}")
print(f"Interpretation: {comparison['interpretation']}")
print(f"Correlation matrix:\n{comparison['correlation_matrix']}")
```

## Modality-Specific Details

### 1. EEG - Temporal Bifurcations

**Detects**: Sudden state changes in electrical activity

**Methods**:
- Sliding window entropy and variance tracking
- Statistical threshold for change detection

**Clinical Applications**:
- Seizure onset detection
- Sleep stage transitions
- Cognitive state changes

**Output**:
```python
{
    'n_bifurcations': 5,
    'bifurcation_points': [1250, 2840, ...],  # Sample indices
    'bifurcation_strengths': [0.72, 0.95, ...],  # 0-1, higher = stronger
    'entropy_trajectory': [...],  # How entropy evolved
    'variance_trajectory': [...]  # How variance evolved
}
```

### 2. MEG - Temporal + Spatial Pattern Bifurcations

**Detects**:
- Temporal bifurcations (like EEG)
- Magnetic field pattern changes (unique to MEG)

**Methods**:
- Global temporal bifurcations (averaged signal)
- Channel-wise bifurcations
- Inter-channel correlation changes (field topology)

**Advantages over EEG**:
- Better spatial resolution
- Detects field pattern reorganization

**Output**:
```python
{
    'global_temporal_bifurcations': 5,
    'global_bifurcation_points': [...],
    'channel_bifurcation_counts': [3, 5, 4, ...],  # Per channel
    'mean_channel_bifurcations': 4.2,
    'spatial_pattern_bifurcations': 3,  # Field topology changes
    'total_bifurcations': 8
}
```

### 3. MRI - Spatial Bifurcations

**Detects**: Structural transitions

**Methods**:
- Gradient: Edges and boundaries
- Laplacian: Curvature changes
- Variance: Texture transitions

**Clinical Applications**:
- Lesion boundary delineation
- Tissue type transitions
- Anatomical structure edges

**Output**:
```python
{
    'bifurcation_map': 2D array,  # Spatial map of transitions
    'n_bifurcation_pixels': 15832,
    'bifurcation_density': 0.042,  # Fraction of pixels
    'mean_bifurcation_strength': 127.3
}
```

### 4. fMRI - Spatio-Temporal Bifurcations ⭐ UNIQUE

**Detects**: Both spatial AND temporal transitions

**Three Types**:
1. **Temporal**: BOLD signal transitions over time
2. **Spatial**: Activation boundary changes
3. **Spatio-temporal**: How activation patterns shift in space-time

**Methods**:
- Temporal: Sliding window on global BOLD
- Spatial: Gradient detection on mean activation
- Spatio-temporal: Pattern correlation changes

**Clinical Applications**:
- Task-related activation shifts
- Functional connectivity changes
- Dynamic network reconfiguration

**Output**:
```python
{
    'temporal_bifurcations': 4,
    'temporal_bifurcation_points': [...],  # Time indices
    'spatial_bifurcations': 12456,  # Activation boundary pixels
    'spatial_bifurcation_density': 0.035,
    'spatiotemporal_bifurcations': 3,  # Pattern shifts
    'spatiotemporal_bifurcation_times': [4.0, 8.2, 12.5],  # seconds
    'pattern_stability': 0.82,  # How stable patterns are
    'total_bifurcations': 12460
}
```

### 5. cMRI - Enhancement + Perfusion Bifurcations

**Detects**: Contrast-specific transitions

**Three Types**:
1. **Enhancement**: Contrast uptake boundaries (static or peak)
2. **Perfusion**: Temporal perfusion transitions (dynamic only)
3. **Vascular**: Vessel branching points

**Methods**:
- Enhancement: Gradient on contrast-enhanced image
- Perfusion: Temporal analysis of contrast curve (4D only)
- Vascular: Laplacian to find vessel bifurcations

**Clinical Applications**:
- Tumor boundary delineation
- Blood-brain barrier breakdown
- Vascular mapping
- Perfusion abnormalities

**Output (static cMRI)**:
```python
{
    'enhancement_bifurcations': 18234,
    'enhancement_bifurcation_density': 0.048,
    'high_intensity_boundary_pixels': 2456,  # Likely tumor edges
    'vascular_bifurcations': 1234,
    'total_bifurcations': 18234
}
```

**Output (dynamic cMRI - 4D)**:
```python
{
    'enhancement_bifurcations': 18234,
    'perfusion_bifurcations': 5,  # Temporal transitions
    'perfusion_transition_times': [...],
    'time_to_peak': 12,  # Time index
    'regional_perfusion_bifurcations': [3, 4, 2, ...],  # By region
    'vascular_bifurcations': 1234,
    'total_bifurcations': 18239
}
```

## Cross-Modal 5x5 Comparison

### Example Output

```python
{
    'modalities': ['EEG', 'MEG', 'MRI', 'fMRI', 'cMRI'],
    'bifurcation_counts': {
        'EEG': 5,
        'MEG': 8,
        'MRI': 15832,
        'fMRI': 12460,
        'cMRI': 18239
    },
    'correlation_matrix': array([
        [1.00, 0.63, 0.00, 0.00, 0.00],  # EEG
        [0.63, 1.00, 0.00, 0.00, 0.00],  # MEG
        [0.00, 0.00, 1.00, 0.79, 0.87],  # MRI
        [0.00, 0.00, 0.79, 1.00, 0.68],  # fMRI
        [0.00, 0.00, 0.87, 0.68, 1.00]   # cMRI
    ]),
    'dominant_modality': 'cMRI',
    'interpretation': "High imbalance: cMRI dominant, others stable",
    'mean_bifurcations': 9308.8,
    'total_bifurcations_all_modalities': 46544
}
```

### Interpretation

**High Correlation Between**:
- EEG ↔ MEG (0.63): Expected - both measure electromagneticactivity
- MRI ↔ cMRI (0.87): Expected - both structural imaging
- MRI ↔ fMRI (0.79): Structure-function relationship

**Low Correlation Between**:
- Electrophysiology (EEG/MEG) ↔ Imaging (MRI/fMRI/cMRI): Different scales

## Clinical Interpretation Examples

### Example 1: Epilepsy Patient

```
EEG: 12 bifurcations (HIGH) → Frequent state changes
MEG: 15 bifurcations → Confirms EEG, adds spatial info
MRI: 18000 bifurcations → Structural lesion detected
Interpretation: Structural epilepsy - lesion causing seizures
```

### Example 2: Tumor Patient

```
EEG: 3 bifurcations (LOW) → Stable electrical activity
fMRI: 5000 spatial bif → Activation around tumor
cMRI: 25000 enhancement bif → Sharp tumor boundary
Interpretation: Well-defined tumor, minimal functional impact
```

### Example 3: Healthy Control

```
EEG: 5 bifurcations → Normal variability
MEG: 7 bifurcations → Consistent with EEG
MRI: 12000 bifurcations → Normal anatomical complexity
fMRI: 8000 bifurcations → Normal activation patterns
cMRI: Not acquired
Interpretation: Balanced complexity across modalities
```

## Advanced Features

### Regional Analysis

fMRI and cMRI support regional bifurcation analysis:
- Divides brain into 8 octants
- Detects bifurcations in each region separately
- Identifies which regions have most transitions

### Temporal Evolution

For dynamic data (fMRI 4D, cMRI 4D):
- Track how bifurcations evolve over time
- Identify critical time points
- Map spatiotemporal patterns

### Method Selection

Each modality supports multiple detection methods:
```python
# MRI: Choose detection method
mri_gradient = detect_bifurcations_by_modality(mri, 'MRI', method='gradient')
mri_laplacian = detect_bifurcations_by_modality(mri, 'MRI', method='laplacian')
mri_variance = detect_bifurcations_by_modality(mri, 'MRI', method='variance')

# fMRI: Choose analysis type
fmri_temporal = detect_bifurcations_by_modality(fmri, 'fMRI', method='temporal')
fmri_spatial = detect_bifurcations_by_modality(fmri, 'fMRI', method='spatial')
fmri_full = detect_bifurcations_by_modality(fmri, 'fMRI', method='spatio-temporal')

# cMRI: Choose what to detect
cmri_enh = detect_bifurcations_by_modality(cmri, 'cMRI', method='enhancement')
cmri_perf = detect_bifurcations_by_modality(cmri, 'cMRI', method='perfusion')
cmri_vasc = detect_bifurcations_by_modality(cmri, 'cMRI', method='vascular')
cmri_all = detect_bifurcations_by_modality(cmri, 'cMRI', method='all')
```

## Summary

✅ **All 5 modalities** now have bifurcation analysis
✅ **Unified interface** via `detect_bifurcations_by_modality()`
✅ **Cross-modal comparison** via `compare_bifurcations_cross_modal()`
✅ **5×5 correlation matrix** for all modality pairs
✅ **Modality-specific methods** tailored to each data type
✅ **Clinical applications** for each modality
✅ **Automatic interpretation** of results

**Every modality you load gets appropriate bifurcation analysis!**
