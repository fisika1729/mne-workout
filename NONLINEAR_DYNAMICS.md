# Non-Linear Dynamics Analysis Guide

## Overview

Non-linear analysis captures complex brain dynamics that **linear correlations miss**. Brain signals are inherently non-linear, exhibiting properties like:
- Chaotic dynamics
- Fractal self-similarity
- Non-linear coupling between regions
- Phase synchronization

This tool provides comprehensive non-linear analysis for MRI/EEG data.

## Why Non-Linear Analysis?

### Limitations of Linear Methods
Linear correlation (Pearson's r) only detects **linear relationships**:
```
y = a*x + b
```

### Brain Dynamics are Non-Linear
Real brain signals show:
- **Threshold effects**: Neurons fire only above threshold
- **Saturation**: Maximum firing rates
- **Oscillatory coupling**: Phase-based interactions
- **Chaotic dynamics**: Sensitive to initial conditions
- **Long-range dependencies**: Fractal temporal structure

### Example
Two signals with **zero linear correlation** can have:
- High mutual information (non-linear dependency)
- Strong phase synchronization
- Directed information flow (transfer entropy)

## Non-Linear Methods Implemented

### 1. Entropy Measures

#### Shannon Entropy
**What it measures**: Information content and unpredictability

**Formula**:
```
H(X) = -Σ p(x) log₂ p(x)
```

**Interpretation**:
- Higher values = More complex, less predictable
- Lower values = More regular, more predictable

**Use in brain signals**:
- Quantifies EEG complexity
- Detects changes in consciousness states
- Indicates information processing capacity

#### Sample Entropy (SampEn)
**What it measures**: Signal regularity without self-matches

**Interpretation**:
- Lower values = More regular patterns
- Higher values = More random/complex
- **Advantages**: Less biased than ApEn, more consistent

**Clinical applications**:
- Alzheimer's: Decreased complexity
- Epilepsy: Reduced entropy before seizures
- Anesthesia: Entropy drops with depth

#### Approximate Entropy (ApEn)
**What it measures**: Regularity with self-matches allowed

**Interpretation**: Similar to SampEn but includes self-matches

### 2. Fractal Dimensions

#### Higuchi Fractal Dimension (HFD)
**What it measures**: Signal complexity and self-similarity

**Range**: 1.0 to 2.0
- **1.0**: Regular, smooth signal
- **1.5**: Typical for brain signals
- **2.0**: Highly complex, random

**Formula**: Based on curve length estimation

**Applications**:
- Seizure prediction (FD increases before seizure)
- Cognitive load (FD increases with task difficulty)
- Brain maturation (FD changes with development)

#### Correlation Dimension
**What it measures**: Attractor dimension in phase space

**Interpretation**:
- Estimates degrees of freedom in the system
- Lower values = More deterministic
- Higher values = More chaotic/random

**Use**: Characterizes underlying dynamical system

### 3. Detrended Fluctuation Analysis (DFA)

**What it measures**: Long-range correlations and scaling behavior

**DFA Exponent (α)**:
- **α < 0.5**: Anti-correlated (mean-reverting)
- **α = 0.5**: Uncorrelated (white noise)
- **α = 1.0**: 1/f noise (pink noise) - typical for brain signals
- **α = 1.5**: Brownian motion (random walk)
- **α > 1.5**: Super-diffusive

**Applications**:
- Detects long-range temporal correlations
- Healthy brain shows 1/f scaling
- Pathology often disrupts this scaling

### 4. Chaos Indicators

#### Lyapunov Exponent
**What it measures**: Sensitivity to initial conditions

**Interpretation**:
- **λ > 0**: Chaotic (exponential divergence)
- **λ = 0**: Stable limit cycle
- **λ < 0**: Stable fixed point

**In brain signals**:
- EEG often shows positive exponent (weak chaos)
- Indicates complex, non-linear dynamics
- Changes with cognitive state

### 5. Coupling Measures (Cross-Signal)

#### Mutual Information (MI)
**What it measures**: Non-linear dependency between signals

**Formula**:
```
MI(X;Y) = H(X) + H(Y) - H(X,Y)
```

**Range**: 0 to ∞
- **0**: Independent signals
- **Higher values**: Stronger dependency

**Advantages over correlation**:
- Detects non-linear relationships
- Captures any form of dependency
- Symmetric measure

**Normalized MI**: Scaled to 0-1 range

#### Phase Synchronization Index (PSI)
**What it measures**: Phase locking between oscillations

**Formula**: Based on Hilbert transform phase analysis

**Range**: 0 to 1
- **0**: No phase synchronization
- **1**: Perfect phase locking

**Computed for each frequency band**:
- Delta, Theta, Alpha, Beta, Gamma

**Applications**:
- Functional connectivity
- Communication between brain regions
- Binding problem in perception

**Advantages**:
- Independent of amplitude
- Sensitive to oscillatory coupling
- Frequency-specific

#### Transfer Entropy (TE)
**What it measures**: **Directed** information flow

**Formula** (simplified):
```
TE(Y→X) = MI(X_future; Y_past | X_past)
```

**Interpretation**:
- Quantifies how much Y helps predict X
- **Asymmetric**: TE(Y→X) ≠ TE(X→Y)
- Higher values = Stronger causal influence

**Applications**:
- Determine directionality (MRI→EEG vs EEG→MRI)
- Identify driver vs driven regions
- Map information flow networks

**In MRI-EEG**:
- Can reveal if structure constrains function
- Or if activity patterns correlate with anatomy

#### Cross-Recurrence Quantification
**What it measures**: Similarity in state-space trajectories

**Method**: Embeds signals in phase space, computes distance matrix

**Recurrence Rate**: Percentage of recurrent states

**Applications**:
- Detects similar dynamical patterns
- Robustto noise
- Non-linear similarity measure

## Usage in the Tool

### Running Non-Linear Analysis

1. **Load Data**:
   - Load EEG data (any supported format)
   - Load MRI data (.nii, .mgz, etc.)

2. **Click "Non-Linear" Button**:
   - Computes all non-linear metrics
   - Analyzes both individual signals and cross-signal metrics
   - Creates 9-panel visualization dashboard

3. **View Results**:
   - Switch to "Non-Linear Dynamics" tab
   - Check "Data Info" tab for detailed metrics
   - Interpret using the guide below

### Visualization Dashboard

The tool creates **9 plots**:

1. **EEG Entropy Measures**: Shannon, Sample, Approximate
2. **Fractal Dimensions**: Higuchi FD, Correlation Dimension
3. **Chaos Indicators**: Lyapunov Exponent, DFA Alpha
4. **Phase Synchronization**: PSI for each frequency band
5. **Mutual Information**: MI and Normalized MI
6. **Transfer Entropy**: Bidirectional flow (MRI↔EEG)
7. **Channel-wise Entropy**: Spatial distribution
8. **Channel-wise FD**: Complexity across channels
9. **Summary**: Automated interpretation

## Interpretation Examples

### Example 1: High Complexity EEG

**Results**:
- Shannon Entropy: 4.5
- Sample Entropy: 1.8
- Higuchi FD: 1.75
- Lyapunov: 0.05

**Interpretation**:
- High information content (Shannon)
- Complex, irregular patterns (SampEn)
- High fractal complexity (FD > 1.5)
- Weakly chaotic dynamics (Lyap > 0)

**Possible causes**:
- Cognitive engagement
- Active information processing
- Awake, alert state

### Example 2: Strong Alpha Synchronization

**Results**:
- PSI Alpha: 0.85
- PSI Delta/Theta/Beta/Gamma: < 0.3
- Mutual Information: 2.1
- Transfer Entropy MRI→EEG: 0.05

**Interpretation**:
- Strong phase locking in alpha band
- Other bands show weak synchronization
- Moderate non-linear coupling (MI)
- Weak directional influence

**Possible meaning**:
- Resting state with eyes closed (alpha dominant)
- Structural-functional correspondence in alpha generators
- Thalamo-cortical synchronization

### Example 3: Chaotic Dynamics

**Results**:
- Lyapunov: 0.15
- Correlation Dimension: 3.5
- DFA Alpha: 1.2
- Transfer Entropy EEG→MRI: 0.12

**Interpretation**:
- Positive Lyapunov = chaotic
- ~3.5 degrees of freedom
- Long-range correlations (DFA > 1.0)
- Information flow from EEG patterns to MRI features

**Implications**:
- Complex, non-linear brain dynamics
- Multiple interacting processes
- Rich temporal structure

### Example 4: Reduced Complexity (Pathological)

**Results**:
- Shannon Entropy: 2.1 (low)
- Sample Entropy: 0.5 (low)
- Higuchi FD: 1.15 (low)
- PSI all bands: < 0.2

**Interpretation**:
- Reduced information content
- Highly regular patterns
- Low complexity
- Weak synchronization

**Possible causes**:
- Deep sleep
- Anesthesia
- Pathology (e.g., coma, severe dementia)
- Artifact/flat signal

## Comparison: Linear vs Non-Linear

| Aspect | Linear (Pearson) | Non-Linear Methods |
|--------|------------------|-------------------|
| Relationships detected | Only linear | Any dependency |
| Phase relationships | Not captured | PSI detects |
| Directionality | None | Transfer Entropy |
| Complexity | Not measured | Multiple metrics |
| Chaos detection | No | Lyapunov exponent |
| Temporal structure | Limited | DFA, fractals |

**Key Insight**: Use **both** linear and non-linear methods for complete analysis!

## Clinical and Research Applications

### Clinical Diagnosis

1. **Epilepsy**:
   - Pre-ictal: Entropy decreases
   - Seizure: Synchronization increases
   - Post-ictal: Complexity gradually returns

2. **Alzheimer's Disease**:
   - Reduced complexity (lower FD, entropy)
   - Decreased long-range correlations (DFA)
   - Loss of non-linear coupling

3. **Anesthesia Monitoring**:
   - Depth correlates with entropy
   - Permutation entropy commonly used
   - Real-time complexity monitoring

4. **Consciousness Assessment**:
   - Complexity measures separate conscious/unconscious
   - PSI indicates network integration
   - Transfer entropy shows information flow

### Research Applications

1. **Cognitive Neuroscience**:
   - Task complexity vs signal complexity
   - Attention effects on synchronization
   - Learning-induced changes in dynamics

2. **Development**:
   - Brain maturation and complexity
   - Network development via coupling measures
   - Critical periods detection

3. **Brain-Computer Interfaces**:
   - Non-linear features for classification
   - Mutual information for feature selection
   - Real-time complexity estimation

## Technical Notes

### Computational Considerations

- **Sample Entropy**: O(N²) - can be slow for long signals
- **Lyapunov Exponent**: Requires long stationary segments
- **Mutual Information**: Sensitive to binning choice
- **Transfer Entropy**: High dimensional, computationally intensive

### Best Practices

1. **Signal Length**:
   - Minimum 1000 samples for entropy
   - Longer for Lyapunov, DFA (10,000+)

2. **Stationarity**:
   - Most methods assume stationarity
   - Use sliding windows for non-stationary signals

3. **Sampling Rate**:
   - Affects fractal dimension estimates
   - Phase synchronization needs adequate sampling

4. **Artifact Removal**:
   - Non-linear metrics sensitive to artifacts
   - Clean data essential

## References

### Key Papers

1. **Entropy**:
   - Richman & Moorman (2000). "Physiological time-series analysis using approximate entropy and sample entropy"

2. **Fractal Dimension**:
   - Higuchi (1988). "Approach to an irregular time series on the basis of the fractal theory"

3. **DFA**:
   - Peng et al. (1994). "Quantification of scaling exponents and crossover phenomena"

4. **Transfer Entropy**:
   - Schreiber (2000). "Measuring information transfer"

5. **Phase Synchronization**:
   - Lachaux et al. (1999). "Measuring phase synchrony in brain signals"

### Books

- Stam (2005). "Nonlinear dynamical analysis of EEG and MEG"
- Kantz & Schreiber (2004). "Nonlinear time series analysis"

## Advantages of This Tool

✓ **Comprehensive**: 15+ non-linear metrics
✓ **Automated**: One-click analysis
✓ **Visual**: 9-panel dashboard
✓ **Interpretable**: Built-in interpretation guide
✓ **Research-grade**: Implements published methods
✓ **Multi-modal**: EEG and MRI analysis
✓ **Channel-wise**: Spatial distribution of complexity
✓ **Cross-signal**: Coupling and directionality

## Limitations

- Computational time for long signals
- Requires clean, artifact-free data
- Parameter choices affect results (e.g., embedding dimension)
- Interpretation requires domain knowledge

## Future Enhancements

Potential additions:
- Permutation entropy
- Multiscale entropy
- Symbolic dynamics
- Wavelet coherence
- Granger causality
- Graph theory metrics
- Time-varying analysis
