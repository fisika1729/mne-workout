# Transform-Based Analysis for Neuroimaging

## Overview

Fourier, Wavelet, and other transforms are **ESSENTIAL** for neuroimaging analysis because they:

1. **Make calculations simpler**: FFT-based correlation is O(n log n) instead of O(n²)
2. **Reveal frequency structure**: Brain oscillations (delta, theta, alpha, beta, gamma)
3. **Detect spectral bifurcations**: State transitions manifest as frequency changes
4. **Enable time-frequency analysis**: Non-stationary signals require wavelets/STFT
5. **Facilitate filtering**: Frequency-domain filter design is straightforward
6. **Cross-modal comparison**: Coherence analysis (EEG ↔ fMRI coupling)

## Why Transforms Are Essential

### 1. **Computational Efficiency**

Direct time-domain convolution: **O(n²)** operations
FFT-based convolution: **O(n log n)** operations

For 10,000 samples:
- Direct: ~100 million operations
- FFT: ~130,000 operations (**~770x faster!**)

### 2. **Brain Oscillations**

Neural activity is inherently oscillatory:
- **Delta (0.5-4 Hz)**: Deep sleep, unconscious processes
- **Theta (4-8 Hz)**: Memory encoding, drowsiness
- **Alpha (8-13 Hz)**: Relaxed wakefulness, closed eyes
- **Beta (13-30 Hz)**: Active thinking, focus, anxiety
- **Gamma (30-50 Hz)**: Cognitive processing, binding

**You CANNOT analyze brain activity without frequency analysis!**

### 3. **Non-Stationary Signals**

Brain signals are non-stationary (statistics change over time). Standard Fourier assumes stationarity → Need time-frequency methods:
- **STFT (Short-Time Fourier Transform)**: Windowed FFT
- **Wavelet Transform**: Adaptive resolution
- **Hilbert-Huang Transform**: Data-adaptive decomposition

### 4. **Spectral Signatures of Bifurcations**

Bifurcations (state transitions) have spectral signatures:
- **Epileptic seizures**: Sudden shift to high-frequency rhythmic activity
- **Sleep stages**: Specific frequency band transitions
- **Anesthesia**: Power shifts from high to low frequencies
- **Cognitive tasks**: Gamma burst during attention

---

## Core Transform Methods

### 1. Power Spectral Density (PSD)

**Why essential**: Shows frequency content and band powers (clinical biomarkers)

```python
from nonlinear_analysis import TransformAnalyzer

# Calculate PSD using Welch's method (robust, recommended)
psd_result = TransformAnalyzer.power_spectral_density(
    eeg_signal,
    sfreq=250,
    method='welch',  # or 'periodogram'
    nperseg=256  # Segment length (longer = better frequency resolution)
)

# Outputs:
# - freqs: Frequency bins
# - psd: Power spectral density
# - total_power: Integrated power
# - band_powers: Power in delta/theta/alpha/beta/gamma
# - dominant_freq: Peak frequency
```

**Clinical Applications**:

```python
# ADHD biomarker: Elevated theta/beta ratio
theta_beta = psd_result['band_powers']['theta'] / psd_result['band_powers']['beta']
if theta_beta > 3.0:
    print("Elevated theta/beta ratio → ADHD signature")

# Alzheimer's: Slowing (increased delta/theta, decreased alpha/beta)
slow_fast_ratio = (psd_result['band_powers']['delta'] + psd_result['band_powers']['theta']) / \
                  (psd_result['band_powers']['alpha'] + psd_result['band_powers']['beta'])
if slow_fast_ratio > 1.5:
    print("Spectral slowing → Possible cognitive impairment")

# Depression: Reduced alpha asymmetry
# (Requires comparing left vs right hemisphere alpha power)
```

**Visualization**:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.semilogy(psd_result['freqs'], psd_result['psd'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('EEG Power Spectrum')
plt.xlim([0, 50])
plt.grid()
plt.show()
```

---

### 2. Wavelet Transform (CWT)

**Why essential**: Time-frequency decomposition for non-stationary signals

Wavelets provide:
- **Time localization**: When did a frequency occur?
- **Frequency localization**: What frequencies are present?
- **Adaptive resolution**: High frequency = good time resolution, low frequency = good frequency resolution

```python
# Continuous Wavelet Transform
wavelet_result = TransformAnalyzer.wavelet_transform(
    eeg_signal,
    sfreq=250,
    wavelet='morl',  # Morlet wavelet (Gaussian-modulated sine)
    scales=None  # Auto-generate scales for 0.5 Hz to Nyquist
)

# Outputs:
# - coefficients: Complex wavelet coefficients (scales x time)
# - freqs: Corresponding frequencies for each scale
# - power: Time-frequency power |coef|² (like spectrogram)
# - time: Time vector
```

**Clinical Applications**:

```python
# Detect spindles in sleep EEG (12-15 Hz transient bursts)
spindle_band_idx = np.logical_and(wavelet_result['freqs'] >= 12,
                                   wavelet_result['freqs'] <= 15)
spindle_power = np.mean(wavelet_result['power'][spindle_band_idx, :], axis=0)

# Threshold for spindle detection
spindle_threshold = np.mean(spindle_power) + 2 * np.std(spindle_power)
spindle_times = wavelet_result['time'][spindle_power > spindle_threshold]

print(f"Detected {len(spindle_times)} sleep spindles")

# Wavelet entropy (measure of signal complexity)
wavelet_entropy = -np.sum(
    wavelet_result['power'] * np.log2(wavelet_result['power'] + 1e-10)
)
```

**Visualization (Scalogram)**:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.pcolormesh(wavelet_result['time'], wavelet_result['freqs'],
               wavelet_result['power'], shading='auto', cmap='jet')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Wavelet Scalogram')
plt.colorbar(label='Power')
plt.ylim([0, 50])
plt.show()
```

**Wavelet Types**:
- **'morl'** (Morlet): Best for general neurophysiology (Gaussian-modulated sine)
- **'mexh'** (Mexican Hat): Good for transient detection (2nd derivative of Gaussian)
- **'cmor'** (Complex Morlet): Provides phase information
- **'db4'** (Daubechies): Discrete wavelet, orthogonal

---

### 3. Short-Time Fourier Transform (STFT)

**Why essential**: Classic time-frequency analysis (spectrogram)

```python
# STFT for spectrogram
stft_result = TransformAnalyzer.short_time_fourier_transform(
    eeg_signal,
    sfreq=250,
    nperseg=256,  # Window length (trade-off: time vs frequency resolution)
    noverlap=128  # 50% overlap (recommended)
)

# Outputs:
# - freqs: Frequency bins
# - times: Time bins
# - spectrogram: STFT magnitude (freqs x times)
# - phase: STFT phase
# - complex: Complex STFT (for reconstruction)
```

**Clinical Applications**:

```python
# Detect ictal onset in epilepsy (rhythmic high-frequency activity)
gamma_band_idx = np.logical_and(stft_result['freqs'] >= 30,
                                 stft_result['freqs'] <= 50)
gamma_power = np.mean(stft_result['spectrogram'][gamma_band_idx, :], axis=0)

# Ictal onset = sustained gamma increase
sustained_gamma = gamma_power > (np.mean(gamma_power) + 3 * np.std(gamma_power))
onset_time = stft_result['times'][sustained_gamma][0] if np.any(sustained_gamma) else None

if onset_time:
    print(f"Ictal onset detected at {onset_time:.2f} seconds")
```

**Visualization**:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.pcolormesh(stft_result['times'], stft_result['freqs'],
               20 * np.log10(stft_result['spectrogram'] + 1e-10),  # dB scale
               shading='auto', cmap='viridis')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram (STFT)')
plt.colorbar(label='Power (dB)')
plt.ylim([0, 50])
plt.show()
```

**Resolution Trade-off**:
- **Longer window (nperseg)**: Better frequency resolution, worse time resolution
- **Shorter window**: Better time resolution, worse frequency resolution
- Typical: 256 samples at 250 Hz = ~1 second window

---

### 4. Coherence Analysis

**Why essential**: Measures frequency-domain correlation between signals (connectivity)

```python
# Coherence between EEG channel 1 and channel 2
coh_result = TransformAnalyzer.coherence_analysis(
    eeg_channel1,
    eeg_channel2,
    sfreq=250,
    nperseg=256
)

# Outputs:
# - freqs: Frequency bins
# - coherence: Coherence values [0, 1] per frequency
# - mean_coherence: Average across all frequencies
# - band_coherence: Coherence per frequency band (delta, theta, alpha, beta, gamma)
```

**Clinical Applications**:

```python
# Functional connectivity in schizophrenia (reduced gamma coherence)
gamma_coherence = coh_result['band_coherence']['gamma']
if gamma_coherence < 0.3:
    print("Reduced gamma-band coherence → Possible schizophrenia")

# Alpha coherence asymmetry in depression
# (Compare coherence between left and right frontal regions)

# Theta coherence increase during memory tasks
theta_coherence = coh_result['band_coherence']['theta']
print(f"Theta coherence: {theta_coherence:.2f}")
```

**Multi-modal Coherence (EEG ↔ fMRI)**:

```python
from nonlinear_analysis import compute_cross_modal_coherence

# Coherence matrix between all modality pairs
coherence_matrix = compute_cross_modal_coherence(
    modality_data_dict={
        'EEG': eeg_signal,
        'MEG': meg_signal,
        'fMRI': fmri_bold_signal
    },
    sfreq_dict={
        'EEG': 250,
        'MEG': 1000,
        'fMRI': 0.5  # TR = 2s → 0.5 Hz
    }
)

print(f"EEG-MEG coherence: {coherence_matrix['coherence_matrix'][0, 1]:.2f}")
print(f"EEG-fMRI coherence: {coherence_matrix['coherence_matrix'][0, 2]:.2f}")
```

**Visualization**:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(coh_result['freqs'], coh_result['coherence'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')
plt.title('EEG Channel 1 ↔ Channel 2 Coherence')
plt.xlim([0, 50])
plt.ylim([0, 1])
plt.grid()
plt.show()
```

---

### 5. Cross-Spectral Density (CSD)

**Why essential**: Complex-valued cross-spectrum provides both magnitude AND phase relationships

```python
# CSD between two signals
csd_result = TransformAnalyzer.cross_spectral_density(
    signal1,
    signal2,
    sfreq=250,
    nperseg=256
)

# Outputs:
# - freqs: Frequency bins
# - csd: Cross-spectral density (complex-valued)
# - magnitude: |CSD|
# - phase: Phase difference between signals
```

**Clinical Applications**:

```python
# Phase-amplitude coupling (PAC) analysis
# Slow phase (theta) modulates fast amplitude (gamma)

# Detect phase lag between regions (connectivity directionality)
phase_lag = csd_result['phase']
mean_phase_lag = np.angle(np.mean(np.exp(1j * phase_lag)))

if np.abs(mean_phase_lag) < 0.1:
    print("Near-zero phase lag → Synchronous activity")
elif mean_phase_lag > 0:
    print(f"Signal 1 leads signal 2 by {mean_phase_lag:.2f} radians")
else:
    print(f"Signal 2 leads signal 1 by {-mean_phase_lag:.2f} radians")
```

---

### 6. Frequency-Domain Bifurcation Detection

**Why essential**: Bifurcations manifest as spectral changes (more reliable than time-domain alone)

```python
# Three methods for frequency-domain bifurcation detection
for method in ['spectral_edge', 'peak_shift', 'bandwidth_change']:
    freq_bif = TransformAnalyzer.frequency_domain_bifurcation_detection(
        eeg_signal,
        sfreq=250,
        method=method
    )

    print(f"\n{method}:")
    print(f"  Bifurcations detected: {freq_bif['n_bifurcations']}")
    print(f"  Times: {freq_bif['bifurcation_times']}")
    print(f"  Scores: {freq_bif['bifurcation_scores']}")
```

**Three Methods**:

1. **Spectral Edge Frequency**: Frequency below which 95% of power lies
   - Bifurcation = sudden shift in spectral edge
   - Use case: Anesthesia depth monitoring (high→low frequencies)

2. **Peak Shift**: Dominant frequency changes
   - Bifurcation = sudden jump in peak frequency
   - Use case: Seizure onset (alpha→theta→gamma transitions)

3. **Bandwidth Change**: Spectral bandwidth variations
   - Bifurcation = narrowing or broadening of spectrum
   - Use case: Cognitive state changes (focused vs diffuse activity)

**Clinical Applications**:

```python
# Combined time + frequency domain bifurcation detection
from nonlinear_analysis import detect_bifurcations_transform_based

result = detect_bifurcations_transform_based(
    eeg_data,  # (channels x time)
    modality_type='EEG',
    sfreq=250
)

# Compare detections
print(f"Time-domain bifurcations: {result['combined_bifurcations']['time_domain_count']}")
print(f"Frequency-domain bifurcations: {result['combined_bifurcations']['frequency_domain_count']}")

# Consensus bifurcations (detected in BOTH domains)
consensus = result['combined_bifurcations']['consensus']
print(f"High-confidence bifurcations (consensus): {len(consensus)}")

# These are the most reliable bifurcations!
```

---

### 7. Hilbert-Huang Transform (HHT)

**Why essential**: Data-adaptive decomposition for non-linear, non-stationary signals

HHT = Empirical Mode Decomposition (EMD) + Hilbert Transform

```python
# Decompose signal into Intrinsic Mode Functions (IMFs)
hht_result = TransformAnalyzer.hilbert_huang_transform(
    eeg_signal,
    sfreq=250,
    n_imfs=5  # Number of IMFs to extract
)

# Outputs:
# - imfs: Intrinsic Mode Functions (oscillatory components)
# - instantaneous_freqs: Frequency variation over time per IMF
# - instantaneous_amps: Amplitude variation over time per IMF
# - time: Time vector
```

**Clinical Applications**:

```python
# Analyze each IMF separately
for i, imf in enumerate(hht_result['imfs']):
    mean_freq = np.mean(hht_result['instantaneous_freqs'][i])
    print(f"IMF {i}: Mean frequency = {mean_freq:.2f} Hz")

# IMF 0 typically contains highest frequency components (gamma)
# IMF 1 contains beta
# IMF 2 contains alpha
# ... and so on

# Detect instantaneous frequency changes (state transitions)
inst_freq = hht_result['instantaneous_freqs'][0]  # First IMF
freq_changes = np.abs(np.diff(inst_freq))
bifurcations = np.where(freq_changes > np.mean(freq_changes) + 2*np.std(freq_changes))[0]
print(f"Detected {len(bifurcations)} instantaneous frequency bifurcations")
```

**Advantages over STFT/Wavelet**:
- **Data-adaptive**: No fixed basis functions
- **Local**: Handles non-stationarity naturally
- **Instantaneous frequency**: True time-varying frequency

**Visualization (Hilbert Spectrum)**:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for i in range(min(3, len(hht_result['imfs']))):  # Plot first 3 IMFs
    plt.subplot(3, 1, i+1)
    plt.scatter(hht_result['time'], hht_result['instantaneous_freqs'][i],
                c=hht_result['instantaneous_amps'][i], s=1, cmap='jet')
    plt.ylabel(f'IMF {i} Freq (Hz)')
    plt.colorbar(label='Amplitude')
    plt.ylim([0, 50])
plt.xlabel('Time (s)')
plt.suptitle('Hilbert Spectrum')
plt.tight_layout()
plt.show()
```

---

### 8. FFT-Based Fast Correlation

**Why essential**: Dramatically faster correlation for long signals

```python
# Fast cross-correlation using FFT
correlation = TransformAnalyzer.fft_based_correlation(signal1, signal2)

# Find lag with maximum correlation
max_lag = np.argmax(correlation)
max_corr = correlation[max_lag]

print(f"Maximum correlation: {max_corr:.2f} at lag {max_lag} samples")
```

**Performance Comparison**:

```python
import time
import numpy as np

signal1 = np.random.randn(10000)
signal2 = np.random.randn(10000)

# Direct correlation (slow)
start = time.time()
direct_corr = np.correlate(signal1, signal2, mode='full')
direct_time = time.time() - start

# FFT-based correlation (fast)
start = time.time()
fft_corr = TransformAnalyzer.fft_based_correlation(signal1, signal2)
fft_time = time.time() - start

print(f"Direct: {direct_time:.4f}s")
print(f"FFT-based: {fft_time:.4f}s")
print(f"Speedup: {direct_time/fft_time:.1f}x")

# Typical speedup: 10-100x for signals > 1000 samples
```

---

## Integrated Transform-Based Analysis

### Unified Interface for All Modalities

```python
from nonlinear_analysis import detect_bifurcations_transform_based

# Automatic transform-based analysis for any modality
result = detect_bifurcations_transform_based(
    eeg_data,
    modality_type='EEG',  # or 'MEG', 'fMRI', 'MRI', 'cMRI'
    sfreq=250
)

# Returns:
# - time_domain: Traditional bifurcation detection
# - frequency_domain: Spectral edge, peak shift, bandwidth methods
# - psd: Power spectral density
# - wavelet: Wavelet transform
# - combined_bifurcations: Consensus detections
```

**Complete Feature Extraction**:

```python
from nonlinear_analysis import compute_transform_features

# Extract all transform-based features
features = compute_transform_features(
    eeg_signal,
    sfreq=250,
    analysis_type='full'  # or 'fast', 'spectral_only', 'wavelet_only'
)

# Clinical biomarkers
print(f"Dominant frequency: {features['dominant_freq']:.2f} Hz")
print(f"Total power: {features['total_power']:.2f}")
print(f"Theta/Beta ratio: {features['theta_beta_ratio']:.2f}")
print(f"Alpha/Theta ratio: {features['alpha_theta_ratio']:.2f}")
print(f"Wavelet entropy: {features['wavelet_entropy']:.2f}")
print(f"Number of IMFs: {features['n_imfs']}")
```

---

## Clinical Workflow Examples

### Example 1: Epilepsy Seizure Detection

```python
from nonlinear_analysis import detect_bifurcations_transform_based, TransformAnalyzer

# Load ictal EEG
eeg_ictal = load_eeg('patient_seizure.set')

# Step 1: Transform-based bifurcation detection
result = detect_bifurcations_transform_based(
    eeg_ictal,
    modality_type='EEG',
    sfreq=256
)

# Step 2: Analyze frequency-domain bifurcations
freq_bif = result['frequency_domain']['spectral_edge']
print(f"Spectral bifurcations: {freq_bif['n_bifurcations']}")

# Step 3: Extract spectral features
features = compute_transform_features(eeg_ictal.mean(axis=0), sfreq=256, analysis_type='fast')

# Seizure signatures:
# - Elevated gamma power
# - Reduced alpha/theta ratio
# - High wavelet entropy

gamma_ratio = features['band_powers']['gamma'] / features['total_power']
if gamma_ratio > 0.3:
    print("⚠ High gamma power → Likely ictal activity")

# Step 4: Wavelet analysis for spindle-like oscillations
wavelet = result['wavelet']
if wavelet is not None:
    spike_freq_band = np.logical_and(wavelet['freqs'] >= 10, wavelet['freqs'] <= 20)
    spike_power = np.mean(wavelet['power'][spike_freq_band, :], axis=0)

    spike_times = wavelet['time'][spike_power > np.percentile(spike_power, 95)]
    print(f"High-amplitude oscillations detected at: {spike_times[:5]}")
```

---

### Example 2: ADHD Diagnosis

```python
# ADHD biomarker: Elevated theta/beta ratio

eeg_resting = load_eeg('patient_resting.set')

# Extract features from frontal channels (Fz, Cz)
fz_signal = eeg_resting[get_channel_index('Fz'), :]

features = compute_transform_features(fz_signal, sfreq=250, analysis_type='spectral_only')

# ADHD threshold: theta/beta > 3.0 (children), > 2.5 (adults)
theta_beta = features['theta_beta_ratio']

print(f"Theta/Beta ratio: {theta_beta:.2f}")

if theta_beta > 3.0:
    print("✓ ADHD biomarker present (elevated theta/beta)")
else:
    print("✗ No ADHD biomarker detected")

# Additional analysis: Band power distribution
band_powers = features['band_powers']
total = sum(band_powers.values())

print("\nBand power distribution:")
for band, power in band_powers.items():
    percentage = 100 * power / total
    print(f"  {band}: {percentage:.1f}%")
```

---

### Example 3: Multi-Modal Coherence (EEG ↔ fMRI)

```python
from nonlinear_analysis import compute_cross_modal_coherence

# Load EEG and fMRI BOLD
eeg_data = load_eeg('simultaneous_eeg.set')
fmri_bold = load_fmri('simultaneous_fmri.nii')

# Extract global signals
eeg_global = eeg_data.mean(axis=0)
fmri_global = fmri_bold.mean(axis=(0, 1, 2))

# Resample to common sampling rate (optional but recommended)
# Here we downsample EEG to fMRI rate

from scipy.signal import resample
n_fmri = len(fmri_global)
eeg_resampled = resample(eeg_global, n_fmri)

# Coherence analysis
coh_result = TransformAnalyzer.coherence_analysis(
    eeg_resampled,
    fmri_global,
    sfreq=0.5,  # fMRI TR = 2s
    nperseg=32
)

print(f"EEG-fMRI mean coherence: {coh_result['mean_coherence']:.2f}")

# Expected coupling: Alpha/Beta EEG ↔ DMN BOLD (anti-correlation)
# Low-frequency BOLD modulated by slow EEG oscillations
```

---

## Best Practices

### 1. **Choosing Transform Methods**

| Signal Type | Recommended Transform | Why |
|-------------|----------------------|-----|
| **Stationary** | FFT (Welch PSD) | Fastest, best for stable signals |
| **Non-stationary** | Wavelet (CWT) or STFT | Time-frequency localization |
| **Non-linear** | Hilbert-Huang (EMD) | Data-adaptive, no assumptions |
| **Connectivity** | Coherence | Frequency-specific coupling |
| **Transient events** | Wavelet (CWT) | Excellent time localization |

### 2. **Parameter Selection**

**PSD (Welch)**:
- `nperseg=256`: Good balance (frequency resolution ~1 Hz at sfreq=250)
- Longer nperseg: Better frequency resolution, worse variance
- Shorter nperseg: Worse frequency resolution, better variance reduction

**Wavelet**:
- `wavelet='morl'`: General-purpose (Gaussian-modulated sine)
- More scales: Better frequency resolution, slower computation
- Fewer scales: Faster, coarser frequency resolution

**STFT**:
- `nperseg=256, noverlap=128`: 50% overlap (recommended)
- Longer window: Better frequency, worse time resolution
- Heisenberg uncertainty: Δt × Δf ≥ constant

### 3. **Frequency Band Selection**

**Standard EEG/MEG bands**:
```python
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50),
    'high_gamma': (50, 100)  # For MEG
}
```

**Clinical adjustments**:
- **Pediatric**: Slower dominant frequencies (adjust bands down by ~2 Hz)
- **Elderly**: Slowing (increased delta/theta, decreased alpha)
- **Pathology**: Custom bands based on observed peaks

### 4. **Interpretation**

**High power in low frequencies (delta/theta)**: Slowing
- Causes: Sleep, drowsiness, dementia, brain damage, anesthesia

**High power in alpha**: Relaxed wakefulness
- Eyes closed: Increased alpha (posterior regions)
- Eyes open: Alpha suppression

**High power in beta**: Active cognition, anxiety
- Task engagement: Increased frontal beta
- Anxiety/stress: Generalized beta increase

**High power in gamma**: Cognitive binding, attention
- Transient gamma bursts: Feature binding
- Sustained gamma: Focused attention

### 5. **Combining Time + Frequency Domains**

**Always use BOTH**:
```python
result = detect_bifurcations_transform_based(data, modality_type='EEG', sfreq=250)

# High-confidence detections: Present in BOTH domains
consensus = result['combined_bifurcations']['consensus']

# Time-domain only: Amplitude changes without spectral signature
time_only = np.setdiff1d(
    result['combined_bifurcations']['time_domain_indices'],
    consensus
)

# Frequency-domain only: Spectral changes without amplitude change
freq_only = np.setdiff1d(
    result['combined_bifurcations']['frequency_domain_indices'],
    consensus
)

print(f"Consensus (high confidence): {len(consensus)}")
print(f"Time-domain only: {len(time_only)}")
print(f"Frequency-domain only: {len(freq_only)}")
```

---

## Summary

Transforms are **ESSENTIAL** for neuroimaging because:

1. **Computational efficiency**: FFT is 10-1000x faster than direct methods
2. **Brain oscillations**: Neural activity is fundamentally oscillatory
3. **Non-stationarity**: Brain signals require time-frequency analysis
4. **Bifurcation detection**: Spectral changes are more reliable than amplitude alone
5. **Connectivity**: Coherence measures frequency-specific coupling
6. **Clinical biomarkers**: Band power ratios, spectral slowing, etc.

This implementation provides:

**9 Transform Methods**:
1. Power Spectral Density (Welch/Periodogram)
2. Wavelet Transform (CWT)
3. Short-Time Fourier Transform (STFT)
4. Coherence Analysis
5. Cross-Spectral Density
6. Frequency-Domain Bifurcation Detection (3 methods)
7. Hilbert-Huang Transform (EMD + Hilbert)
8. FFT-Based Fast Correlation
9. Integrated multi-modal coherence

**Unified Interface**:
- `detect_bifurcations_transform_based()`: Automatic transform-based analysis for all 5 modalities
- `compute_transform_features()`: Extract all spectral features
- `compute_cross_modal_coherence()`: Multi-modal frequency-domain coupling

**Clinical Applications**:
- Epilepsy: Seizure detection via spectral bifurcations
- ADHD: Theta/beta ratio biomarker
- Alzheimer's: Spectral slowing
- Schizophrenia: Reduced gamma coherence
- Depression: Alpha asymmetry
- Sleep staging: Frequency band transitions
- Anesthesia: Spectral edge monitoring

**Use transforms for ALL neuroimaging analysis to get efficient, robust, clinically-meaningful results!**
