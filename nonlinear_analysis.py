#!/usr/bin/env python3
"""
Non-linear Dynamics Analysis Module
Provides methods for analyzing non-linear relationships in neuroimaging data
Including bifurcation detection and analysis
"""

import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import sobel, generic_gradient_magnitude
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class NonLinearAnalyzer:
    """
    Comprehensive non-linear dynamics analysis for brain data
    """

    @staticmethod
    def mutual_information(x, y, bins=20):
        """
        Calculate mutual information between two signals

        MI measures non-linear dependencies that linear correlation might miss

        Parameters:
        -----------
        x, y : array-like
            Input signals
        bins : int
            Number of bins for histogram

        Returns:
        --------
        mi : float
            Mutual information value
        """
        # Discretize continuous variables
        x_discrete = np.digitize(x, bins=np.linspace(x.min(), x.max(), bins))
        y_discrete = np.digitize(y, bins=np.linspace(y.min(), y.max(), bins))

        # Calculate mutual information
        mi = mutual_info_score(x_discrete, y_discrete)

        return mi

    @staticmethod
    def normalized_mutual_information(x, y, bins=20):
        """
        Calculate normalized mutual information (0 to 1)

        Normalized by the geometric mean of individual entropies
        """
        mi = NonLinearAnalyzer.mutual_information(x, y, bins)

        # Calculate entropies
        h_x = NonLinearAnalyzer.shannon_entropy(x, bins)
        h_y = NonLinearAnalyzer.shannon_entropy(y, bins)

        # Normalize
        if h_x * h_y > 0:
            nmi = mi / np.sqrt(h_x * h_y)
        else:
            nmi = 0

        return nmi

    @staticmethod
    def shannon_entropy(x, bins=20):
        """
        Calculate Shannon entropy

        Measures information content/complexity of signal

        Parameters:
        -----------
        x : array-like
            Input signal
        bins : int
            Number of bins for histogram

        Returns:
        --------
        entropy : float
            Shannon entropy value
        """
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros

        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return entropy

    @staticmethod
    def sample_entropy(x, m=2, r=None):
        """
        Calculate Sample Entropy (SampEn)

        Measures complexity and regularity of time series
        Lower values = more regular/predictable
        Higher values = more complex/random

        Parameters:
        -----------
        x : array-like
            Input time series
        m : int
            Embedding dimension (default: 2)
        r : float
            Tolerance (default: 0.2 * std(x))

        Returns:
        --------
        sampen : float
            Sample entropy value
        """
        x = np.array(x)
        N = len(x)

        if r is None:
            r = 0.2 * np.std(x)

        # Create templates of length m and m+1
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([x[i:i+m] for i in range(N - m + 1)])
            count = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j:
                        if _maxdist(patterns[i], patterns[j]) <= r:
                            count += 1
            return count / (N - m + 1)

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        if phi_m > 0 and phi_m1 > 0:
            sampen = -np.log(phi_m1 / phi_m)
        else:
            sampen = np.inf

        return sampen

    @staticmethod
    def approximate_entropy(x, m=2, r=None):
        """
        Calculate Approximate Entropy (ApEn)

        Similar to Sample Entropy but includes self-matches
        Measures regularity and unpredictability

        Parameters:
        -----------
        x : array-like
            Input time series
        m : int
            Embedding dimension
        r : float
            Tolerance

        Returns:
        --------
        apen : float
            Approximate entropy value
        """
        x = np.array(x)
        N = len(x)

        if r is None:
            r = 0.2 * np.std(x)

        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([x[i:i+m] for i in range(N - m + 1)])
            C = np.zeros(len(patterns))

            for i in range(len(patterns)):
                matches = 0
                for j in range(len(patterns)):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        matches += 1
                C[i] = matches / (N - m + 1)

            phi = np.mean(np.log(C + 1e-10))
            return phi

        apen = _phi(m) - _phi(m + 1)

        return apen

    @staticmethod
    def higuchi_fractal_dimension(x, kmax=10):
        """
        Calculate Higuchi Fractal Dimension

        Measures complexity and self-similarity of time series
        Higher values indicate more complex signals

        Parameters:
        -----------
        x : array-like
            Input time series
        kmax : int
            Maximum k value

        Returns:
        --------
        hfd : float
            Higuchi fractal dimension
        """
        x = np.array(x)
        N = len(x)

        L = []
        k_values = range(1, kmax + 1)

        for k in k_values:
            Lk = []
            for m in range(k):
                Lmk = 0
                max_idx = int(np.floor((N - m - 1) / k))

                for i in range(1, max_idx + 1):
                    Lmk += abs(x[m + i * k] - x[m + (i - 1) * k])

                if max_idx > 0:
                    Lmk = (Lmk * (N - 1)) / (max_idx * k * k)
                    Lk.append(Lmk)

            if len(Lk) > 0:
                L.append(np.mean(Lk))

        # Fit log-log plot
        if len(L) > 1:
            log_k = np.log(list(k_values[:len(L)]))
            log_L = np.log(L)

            # Linear regression
            slope, _ = np.polyfit(log_k, log_L, 1)
            hfd = -slope
        else:
            hfd = 0

        return hfd

    @staticmethod
    def detrended_fluctuation_analysis(x, min_scale=4, max_scale=None, num_scales=10):
        """
        Detrended Fluctuation Analysis (DFA)

        Measures long-range correlations and self-similarity
        DFA exponent (alpha):
        - 0.5: uncorrelated (white noise)
        - 1.0: 1/f noise (pink noise)
        - 1.5: Brownian noise

        Parameters:
        -----------
        x : array-like
            Input time series
        min_scale : int
            Minimum window size
        max_scale : int
            Maximum window size (default: len(x)/4)
        num_scales : int
            Number of scales to test

        Returns:
        --------
        alpha : float
            DFA exponent
        """
        x = np.array(x)
        N = len(x)

        if max_scale is None:
            max_scale = N // 4

        # Create cumulative sum (integration)
        y = np.cumsum(x - np.mean(x))

        # Generate scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale),
                            num_scales, dtype=int)
        scales = np.unique(scales)

        F = []

        for scale in scales:
            # Divide into segments
            num_segments = N // scale

            if num_segments < 1:
                continue

            fluctuations = []

            for v in range(num_segments):
                # Extract segment
                segment = y[v * scale:(v + 1) * scale]

                # Fit polynomial trend
                t = np.arange(len(segment))
                coeffs = np.polyfit(t, segment, 1)
                trend = np.polyval(coeffs, t)

                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
                fluctuations.append(fluctuation)

            if len(fluctuations) > 0:
                F.append(np.mean(fluctuations))

        # Fit log-log plot
        if len(F) > 1:
            log_scales = np.log(scales[:len(F)])
            log_F = np.log(F)

            # Linear regression to get alpha
            alpha, _ = np.polyfit(log_scales, log_F, 1)
        else:
            alpha = 0

        return alpha

    @staticmethod
    def phase_synchronization_index(x, y, freq_band=None, sfreq=250):
        """
        Calculate Phase Synchronization Index (PSI)

        Measures synchronization between oscillatory signals
        Based on phase locking value

        Parameters:
        -----------
        x, y : array-like
            Input signals
        freq_band : tuple
            Frequency band (fmin, fmax) for filtering
        sfreq : float
            Sampling frequency

        Returns:
        --------
        psi : float
            Phase synchronization index (0 to 1)
        """
        x = np.array(x)
        y = np.array(y)

        # Filter to frequency band if specified
        if freq_band is not None:
            sos = signal.butter(4, freq_band, btype='bandpass',
                               fs=sfreq, output='sos')
            x = signal.sosfilt(sos, x)
            y = signal.sosfilt(sos, y)

        # Calculate analytic signal using Hilbert transform
        analytic_x = signal.hilbert(x)
        analytic_y = signal.hilbert(y)

        # Extract phases
        phase_x = np.angle(analytic_x)
        phase_y = np.angle(analytic_y)

        # Calculate phase difference
        phase_diff = phase_x - phase_y

        # Calculate phase locking value (PLV)
        psi = np.abs(np.mean(np.exp(1j * phase_diff)))

        return psi

    @staticmethod
    def cross_recurrence_quantification(x, y, embedding_dim=2, delay=1, threshold=None):
        """
        Cross-Recurrence Quantification Analysis

        Measures similarity in state-space trajectories

        Parameters:
        -----------
        x, y : array-like
            Input time series
        embedding_dim : int
            Embedding dimension
        delay : int
            Time delay
        threshold : float
            Recurrence threshold

        Returns:
        --------
        metrics : dict
            Recurrence rate, determinism, etc.
        """
        # Time-delay embedding
        def embed(signal, m, tau):
            N = len(signal)
            M = N - (m - 1) * tau
            embedded = np.zeros((M, m))
            for i in range(M):
                embedded[i] = signal[i:i + m * tau:tau]
            return embedded

        x_embedded = embed(x, embedding_dim, delay)
        y_embedded = embed(y, embedding_dim, delay)

        # Calculate cross-recurrence matrix
        distances = np.zeros((len(x_embedded), len(y_embedded)))
        for i in range(len(x_embedded)):
            for j in range(len(y_embedded)):
                distances[i, j] = np.linalg.norm(x_embedded[i] - y_embedded[j])

        # Set threshold
        if threshold is None:
            threshold = 0.1 * np.max(distances)

        # Create recurrence matrix
        recurrence_matrix = (distances < threshold).astype(int)

        # Calculate metrics
        recurrence_rate = np.sum(recurrence_matrix) / recurrence_matrix.size

        metrics = {
            'recurrence_rate': recurrence_rate,
            'recurrence_matrix': recurrence_matrix
        }

        return metrics

    @staticmethod
    def transfer_entropy(x, y, k=1, l=1):
        """
        Simplified Transfer Entropy

        Measures directed information flow from y to x
        Captures non-linear causal relationships

        Parameters:
        -----------
        x, y : array-like
            Source (y) and target (x) time series
        k : int
            History length for x
        l : int
            History length for y

        Returns:
        --------
        te : float
            Transfer entropy value
        """
        x = np.array(x)
        y = np.array(y)

        # Discretize signals
        bins = 10
        x_discrete = np.digitize(x, bins=np.linspace(x.min(), x.max(), bins))
        y_discrete = np.digitize(y, bins=np.linspace(y.min(), y.max(), bins))

        N = min(len(x_discrete), len(y_discrete))
        max_lag = max(k, l)

        if N <= max_lag + 1:
            return 0

        # Build state vectors (simplified version)
        x_present = x_discrete[max_lag:N]
        x_past = x_discrete[max_lag-k:N-k]
        y_past = y_discrete[max_lag-l:N-l]

        # Calculate conditional mutual information
        # TE = I(X_present; Y_past | X_past)
        # Simplified calculation using mutual information

        # This is a simplified approximation
        mi_xy = NonLinearAnalyzer.mutual_information(
            np.column_stack([x_present, x_past]).flatten(),
            y_past, bins=bins
        )
        mi_x = NonLinearAnalyzer.mutual_information(
            x_present, x_past, bins=bins
        )

        te = mi_xy - mi_x

        return max(0, te)  # Transfer entropy is non-negative

    @staticmethod
    def correlation_dimension(x, max_dim=10, r_range=(0.01, 0.5), num_r=20):
        """
        Calculate Correlation Dimension

        Estimates the fractal dimension of attractor
        Measure of system complexity

        Parameters:
        -----------
        x : array-like
            Input time series
        max_dim : int
            Maximum embedding dimension
        r_range : tuple
            Range of radii to test
        num_r : int
            Number of radii

        Returns:
        --------
        cd : float
            Correlation dimension
        """
        x = np.array(x)

        # Normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-10)

        # Create embedded vectors for maximum dimension
        N = len(x)
        tau = 1
        M = N - (max_dim - 1) * tau

        if M < 10:
            return 0

        embedded = np.zeros((M, max_dim))
        for i in range(M):
            embedded[i] = x[i:i + max_dim * tau:tau]

        # Calculate pairwise distances
        distances = pdist(embedded, metric='euclidean')

        # Range of radii
        r_values = np.logspace(np.log10(r_range[0] * np.std(x)),
                              np.log10(r_range[1] * np.std(x)),
                              num_r)

        # Calculate correlation integral for each radius
        C_r = []
        for r in r_values:
            C_r.append(np.sum(distances < r) / len(distances))

        # Estimate dimension from log-log slope
        log_r = np.log(r_values)
        log_C = np.log(np.array(C_r) + 1e-10)

        # Find linear region and fit
        valid_idx = np.isfinite(log_C) & (log_C != 0)
        if np.sum(valid_idx) > 2:
            cd, _ = np.polyfit(log_r[valid_idx], log_C[valid_idx], 1)
        else:
            cd = 0

        return cd

    @staticmethod
    def lyapunov_exponent_rosenstein(x, emb_dim=10, tau=1, mean_period=None):
        """
        Estimate largest Lyapunov exponent using Rosenstein's method

        Positive exponent indicates chaotic behavior

        Parameters:
        -----------
        x : array-like
            Input time series
        emb_dim : int
            Embedding dimension
        tau : int
            Time delay
        mean_period : int
            Mean period of the signal (estimated if None)

        Returns:
        --------
        lambda_max : float
            Largest Lyapunov exponent
        """
        x = np.array(x)
        N = len(x)

        # Time-delay embedding
        M = N - (emb_dim - 1) * tau
        if M < 10:
            return 0

        embedded = np.zeros((M, emb_dim))
        for i in range(M):
            embedded[i] = x[i:i + emb_dim * tau:tau]

        # Estimate mean period if not provided
        if mean_period is None:
            # Use autocorrelation to estimate
            autocorr = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            # Find first minimum
            peaks = signal.find_peaks(-autocorr)[0]
            if len(peaks) > 0:
                mean_period = peaks[0]
            else:
                mean_period = 10

        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(embedded)
        distances, indices = nbrs.kneighbors(embedded)

        # Calculate divergence
        max_iter = min(M - mean_period, 100)
        divergence = []

        for k in range(max_iter):
            dist_sum = 0
            count = 0

            for i in range(M - k):
                j = indices[i, 1]  # Nearest neighbor

                # Check if we can follow the trajectory
                if i + k < M and j + k < M:
                    # Calculate distance after k steps
                    d = np.linalg.norm(embedded[i + k] - embedded[j + k])
                    if d > 0:
                        dist_sum += np.log(d)
                        count += 1

            if count > 0:
                divergence.append(dist_sum / count)

        # Fit linear region to get Lyapunov exponent
        if len(divergence) > 2:
            t = np.arange(len(divergence))
            # Fit first 1/3 to 2/3 of the data
            fit_start = len(divergence) // 3
            fit_end = 2 * len(divergence) // 3

            if fit_end > fit_start + 2:
                lambda_max, _ = np.polyfit(t[fit_start:fit_end],
                                          divergence[fit_start:fit_end], 1)
            else:
                lambda_max = 0
        else:
            lambda_max = 0

        return lambda_max

    @staticmethod
    def zero_one_test_for_chaos(x):
        """
        0-1 Test for Chaos (Gottwald & Melbourne)

        Detects chaotic behavior and bifurcations
        Returns value close to 0 for regular dynamics, close to 1 for chaotic

        Parameters:
        -----------
        x : array-like
            Input time series

        Returns:
        --------
        K : float
            Test statistic (0 = regular, 1 = chaotic)
        """
        x = np.array(x)
        N = len(x)

        # Choose random phase parameter
        c = np.random.uniform(np.pi/5, 4*np.pi/5)

        # Compute translation variables
        p = np.zeros(N)
        q = np.zeros(N)

        for n in range(1, N):
            p[n] = p[n-1] + x[n] * np.cos(n * c)
            q[n] = q[n-1] + x[n] * np.sin(n * c)

        # Compute mean square displacement
        M_c = np.zeros(int(N / 10))

        for n in range(1, int(N / 10)):
            diff_p = p[n:] - p[:-n]
            diff_q = q[n:] - q[:-n]
            M_c[n] = np.mean(diff_p**2 + diff_q**2)

        # Compute correlation coefficient
        n_vals = np.arange(1, int(N / 10))
        if len(n_vals) > 1 and len(M_c[1:]) > 1:
            corr = np.corrcoef(n_vals, M_c[1:])[0, 1]
            K = corr
        else:
            K = 0

        # Normalize to [0, 1]
        K = abs(K)

        return K

    @staticmethod
    def detect_temporal_bifurcations(x, window_size=None, overlap=0.5):
        """
        Detect bifurcation points in temporal signals (EEG)

        Uses sliding window analysis to detect sudden changes in dynamics

        Parameters:
        -----------
        x : array-like
            Time series (EEG signal)
        window_size : int
            Window size for analysis (default: len(x)//10)
        overlap : float
            Overlap between windows (0-1)

        Returns:
        --------
        bifurcations : dict
            Contains bifurcation points, metrics, and locations
        """
        x = np.array(x)
        N = len(x)

        if window_size is None:
            window_size = max(100, N // 10)

        step = int(window_size * (1 - overlap))

        # Metrics to track
        entropies = []
        lyapunovs = []
        variances = []
        positions = []

        # Sliding window analysis
        for i in range(0, N - window_size, step):
            window = x[i:i + window_size]
            positions.append(i + window_size // 2)

            # Compute metrics
            try:
                entropy = NonLinearAnalyzer.shannon_entropy(window)
                entropies.append(entropy)
            except:
                entropies.append(0)

            variances.append(np.var(window))

            # Simplified Lyapunov (too expensive for all windows)
            # Just use variance rate of change as proxy
            if len(variances) > 1:
                lyap_proxy = abs(variances[-1] - variances[-2])
            else:
                lyap_proxy = 0
            lyapunovs.append(lyap_proxy)

        # Detect sudden changes (bifurcation points)
        entropies = np.array(entropies)
        variances = np.array(variances)
        positions = np.array(positions)

        # Compute derivatives
        if len(entropies) > 2:
            entropy_diff = np.abs(np.diff(entropies))
            variance_diff = np.abs(np.diff(variances))

            # Normalize
            entropy_diff = entropy_diff / (np.max(entropy_diff) + 1e-10)
            variance_diff = variance_diff / (np.max(variance_diff) + 1e-10)

            # Combined change metric
            change_metric = entropy_diff + variance_diff

            # Find peaks (bifurcation candidates)
            threshold = np.mean(change_metric) + 2 * np.std(change_metric)
            bifurcation_indices = np.where(change_metric > threshold)[0]

            bifurcation_times = positions[bifurcation_indices]
            bifurcation_strengths = change_metric[bifurcation_indices]
        else:
            bifurcation_times = np.array([])
            bifurcation_strengths = np.array([])

        return {
            'bifurcation_points': bifurcation_times,
            'bifurcation_strengths': bifurcation_strengths,
            'entropy_trajectory': entropies,
            'variance_trajectory': variances,
            'positions': positions,
            'n_bifurcations': len(bifurcation_times)
        }

    @staticmethod
    def detect_spatial_bifurcations(img, method='gradient'):
        """
        Detect bifurcation points in spatial data (MRI images)

        Identifies structural transitions, boundaries, and discontinuities

        Parameters:
        -----------
        img : ndarray
            2D or 3D image (MRI data)
        method : str
            'gradient', 'laplacian', or 'variance'

        Returns:
        --------
        bifurcations : dict
            Spatial bifurcation map and metrics
        """
        img = np.array(img)

        if len(img.shape) == 3:
            # For 3D, analyze middle slice or average
            img_2d = img[:, :, img.shape[2] // 2]
        else:
            img_2d = img

        if method == 'gradient':
            # Compute gradient magnitude (edges/boundaries)
            gx = sobel(img_2d, axis=0, mode='constant')
            gy = sobel(img_2d, axis=1, mode='constant')
            gradient_mag = np.sqrt(gx**2 + gy**2)

            bifurcation_map = gradient_mag

        elif method == 'laplacian':
            # Laplacian for curvature changes
            from scipy.ndimage import laplace
            laplacian = laplace(img_2d)
            bifurcation_map = np.abs(laplacian)

        elif method == 'variance':
            # Local variance (texture changes)
            from scipy.ndimage import generic_filter
            def local_var(values):
                return np.var(values)

            bifurcation_map = generic_filter(img_2d, local_var, size=5)

        # Threshold to find significant bifurcations
        threshold = np.mean(bifurcation_map) + 2 * np.std(bifurcation_map)
        significant_bifurcations = bifurcation_map > threshold

        # Find bifurcation locations
        bifurcation_coords = np.where(significant_bifurcations)

        return {
            'bifurcation_map': bifurcation_map,
            'significant_bifurcations': significant_bifurcations,
            'bifurcation_coordinates': bifurcation_coords,
            'n_bifurcation_pixels': np.sum(significant_bifurcations),
            'mean_bifurcation_strength': np.mean(bifurcation_map[significant_bifurcations])
                                         if np.any(significant_bifurcations) else 0,
            'bifurcation_density': np.sum(significant_bifurcations) / bifurcation_map.size
        }

    @staticmethod
    def compare_bifurcations_temporal_spatial(temporal_bif, spatial_bif, temporal_signal, spatial_image):
        """
        Compare bifurcations between temporal (EEG) and spatial (MRI) data

        Correlates temporal transition points with spatial discontinuities

        Parameters:
        -----------
        temporal_bif : dict
            Output from detect_temporal_bifurcations
        spatial_bif : dict
            Output from detect_spatial_bifurcations
        temporal_signal : array
            Original EEG signal
        spatial_image : array
            Original MRI image

        Returns:
        --------
        correlation : dict
            Correlation metrics between temporal and spatial bifurcations
        """
        comparison = {}

        # Number of bifurcations
        comparison['n_temporal_bifurcations'] = temporal_bif['n_bifurcations']
        comparison['n_spatial_bifurcation_regions'] = spatial_bif['n_bifurcation_pixels']
        comparison['spatial_bifurcation_density'] = spatial_bif['bifurcation_density']

        # Temporal bifurcation density
        if len(temporal_signal) > 0:
            comparison['temporal_bifurcation_density'] = (
                temporal_bif['n_bifurcations'] / len(temporal_signal)
            )
        else:
            comparison['temporal_bifurcation_density'] = 0

        # Complexity correlation
        # Higher bifurcations = higher complexity
        temporal_complexity = temporal_bif['n_bifurcations']
        spatial_complexity = spatial_bif['n_bifurcation_pixels']

        # Normalize
        if temporal_complexity > 0 or spatial_complexity > 0:
            max_val = max(temporal_complexity, spatial_complexity)
            temporal_norm = temporal_complexity / max_val
            spatial_norm = spatial_complexity / max_val

            # Simple correlation metric
            comparison['complexity_ratio'] = (
                min(temporal_norm, spatial_norm) / (max(temporal_norm, spatial_norm) + 1e-10)
            )
        else:
            comparison['complexity_ratio'] = 0

        # Bifurcation strength correlation
        if temporal_bif['n_bifurcations'] > 0 and spatial_bif['n_bifurcation_pixels'] > 0:
            mean_temporal_strength = np.mean(temporal_bif['bifurcation_strengths'])
            mean_spatial_strength = spatial_bif['mean_bifurcation_strength']

            # Normalize and compare
            total_strength = mean_temporal_strength + mean_spatial_strength
            if total_strength > 0:
                comparison['strength_balance'] = mean_temporal_strength / total_strength
            else:
                comparison['strength_balance'] = 0.5
        else:
            comparison['strength_balance'] = 0.5

        # Interpretation
        if comparison['complexity_ratio'] > 0.7:
            comparison['interpretation'] = "High correlation: Temporal and spatial bifurcations are balanced"
        elif comparison['temporal_bifurcation_density'] > comparison['spatial_bifurcation_density']:
            comparison['interpretation'] = "Temporal dominant: More dynamic transitions in EEG than structural in MRI"
        else:
            comparison['interpretation'] = "Spatial dominant: More structural complexity in MRI than temporal in EEG"

        return comparison

    @staticmethod
    def detect_meg_bifurcations(meg_data, sfreq=1000, window_size=None, overlap=0.5):
        """
        Detect bifurcations in MEG data

        MEG has higher spatial resolution than EEG, so we can detect:
        - Temporal bifurcations (like EEG)
        - Sensor-specific bifurcations
        - Field pattern bifurcations

        Parameters:
        -----------
        meg_data : array-like
            MEG data (channels x time)
        sfreq : float
            Sampling frequency
        window_size : int
            Window size for analysis
        overlap : float
            Overlap between windows

        Returns:
        --------
        bifurcations : dict
            MEG bifurcation metrics
        """
        if len(meg_data.shape) == 1:
            # Single channel
            meg_data = meg_data.reshape(1, -1)

        n_channels, n_samples = meg_data.shape

        # Global temporal bifurcations (average across channels)
        global_signal = np.mean(meg_data, axis=0)
        global_bif = NonLinearAnalyzer.detect_temporal_bifurcations(
            global_signal, window_size, overlap
        )

        # Channel-wise bifurcations
        channel_bifurcations = []
        for ch_idx in range(min(n_channels, 20)):  # Limit to 20 channels
            ch_data = meg_data[ch_idx, :]
            try:
                ch_bif = NonLinearAnalyzer.detect_temporal_bifurcations(
                    ch_data, window_size, overlap
                )
                channel_bifurcations.append(ch_bif['n_bifurcations'])
            except:
                channel_bifurcations.append(0)

        # Spatial pattern changes (field topology bifurcations)
        # Compute correlation between channels over time
        if window_size is None:
            window_size = max(100, n_samples // 10)

        step = int(window_size * (1 - overlap))
        spatial_pattern_changes = []

        for i in range(0, n_samples - window_size, step):
            window = meg_data[:, i:i + window_size]
            # Compute inter-channel correlation
            corr_matrix = np.corrcoef(window)
            # Flatten upper triangle
            spatial_pattern = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            spatial_pattern_changes.append(np.mean(spatial_pattern))

        # Detect changes in spatial patterns
        spatial_pattern_changes = np.array(spatial_pattern_changes)
        if len(spatial_pattern_changes) > 2:
            pattern_diff = np.abs(np.diff(spatial_pattern_changes))
            pattern_diff_norm = pattern_diff / (np.max(pattern_diff) + 1e-10)

            threshold = np.mean(pattern_diff_norm) + 2 * np.std(pattern_diff_norm)
            spatial_bifurcations = np.where(pattern_diff_norm > threshold)[0]
        else:
            spatial_bifurcations = np.array([])

        return {
            'global_temporal_bifurcations': global_bif['n_bifurcations'],
            'global_bifurcation_points': global_bif['bifurcation_points'],
            'global_bifurcation_strengths': global_bif['bifurcation_strengths'],
            'channel_bifurcation_counts': channel_bifurcations,
            'mean_channel_bifurcations': np.mean(channel_bifurcations),
            'spatial_pattern_bifurcations': len(spatial_bifurcations),
            'spatial_pattern_change_points': spatial_bifurcations,
            'total_bifurcations': global_bif['n_bifurcations'] + len(spatial_bifurcations),
            'modality': 'MEG'
        }

    @staticmethod
    def detect_fmri_bifurcations(fmri_data, tr=2.0, method='spatio-temporal'):
        """
        Detect bifurcations in fMRI data

        fMRI is unique: 4D data (x, y, z, time)
        Can detect:
        - Temporal bifurcations: BOLD signal transitions over time
        - Spatial bifurcations: Activation boundary changes
        - Spatio-temporal bifurcations: How activation patterns shift

        Parameters:
        -----------
        fmri_data : ndarray
            4D fMRI data (x, y, z, time) or 3D (averaged over time)
        tr : float
            Repetition time (seconds)
        method : str
            'temporal', 'spatial', or 'spatio-temporal'

        Returns:
        --------
        bifurcations : dict
            fMRI bifurcation metrics
        """
        if len(fmri_data.shape) == 4:
            x, y, z, t = fmri_data.shape
            has_temporal = True
        elif len(fmri_data.shape) == 3:
            x, y, z = fmri_data.shape
            has_temporal = False
            t = 1
        else:
            raise ValueError("fMRI data must be 3D or 4D")

        results = {'modality': 'fMRI'}

        # Temporal bifurcations (if time dimension exists)
        if has_temporal and method in ['temporal', 'spatio-temporal']:
            # Average BOLD signal across space
            global_bold = np.mean(fmri_data, axis=(0, 1, 2))

            temporal_bif = NonLinearAnalyzer.detect_temporal_bifurcations(
                global_bold, window_size=min(10, t // 3), overlap=0.5
            )

            results['temporal_bifurcations'] = temporal_bif['n_bifurcations']
            results['temporal_bifurcation_points'] = temporal_bif['bifurcation_points']
            results['temporal_bifurcation_strengths'] = temporal_bif['bifurcation_strengths']

            # Regional temporal bifurcations
            # Divide into 8 regions and detect bifurcations in each
            regions = NonLinearAnalyzer._divide_fmri_regions(fmri_data)
            regional_temporal_bif = []

            for region in regions:
                if region.size > 0:
                    region_timeseries = np.mean(region, axis=(0, 1, 2))
                    try:
                        reg_bif = NonLinearAnalyzer.detect_temporal_bifurcations(
                            region_timeseries, window_size=min(10, len(region_timeseries) // 3)
                        )
                        regional_temporal_bif.append(reg_bif['n_bifurcations'])
                    except:
                        regional_temporal_bif.append(0)

            results['regional_temporal_bifurcations'] = regional_temporal_bif
            results['mean_regional_bifurcations'] = np.mean(regional_temporal_bif)
        else:
            results['temporal_bifurcations'] = 0

        # Spatial bifurcations (activation boundaries)
        if method in ['spatial', 'spatio-temporal']:
            if has_temporal:
                # Use mean activation map
                mean_activation = np.mean(fmri_data, axis=3)
            else:
                mean_activation = fmri_data

            # Detect spatial bifurcations on middle slice
            middle_slice = mean_activation[:, :, z // 2]
            spatial_bif = NonLinearAnalyzer.detect_spatial_bifurcations(
                middle_slice, method='gradient'
            )

            results['spatial_bifurcations'] = spatial_bif['n_bifurcation_pixels']
            results['spatial_bifurcation_density'] = spatial_bif['bifurcation_density']
            results['spatial_bifurcation_map'] = spatial_bif['bifurcation_map']

        # Spatio-temporal bifurcations (activation pattern changes)
        if has_temporal and method == 'spatio-temporal':
            # Track how activation patterns change over time
            pattern_correlations = []

            for time_idx in range(t - 1):
                corr = np.corrcoef(
                    fmri_data[:, :, :, time_idx].flatten(),
                    fmri_data[:, :, :, time_idx + 1].flatten()
                )[0, 1]
                pattern_correlations.append(corr)

            pattern_correlations = np.array(pattern_correlations)

            # Detect sudden pattern changes
            pattern_changes = 1 - pattern_correlations  # Low corr = big change
            threshold = np.mean(pattern_changes) + 2 * np.std(pattern_changes)
            spatiotemporal_bif = np.where(pattern_changes > threshold)[0]

            results['spatiotemporal_bifurcations'] = len(spatiotemporal_bif)
            results['spatiotemporal_bifurcation_times'] = spatiotemporal_bif * tr
            results['pattern_stability'] = np.mean(pattern_correlations)

        # Total bifurcations
        total = results.get('temporal_bifurcations', 0) + results.get('spatial_bifurcations', 0)
        results['total_bifurcations'] = total

        return results

    @staticmethod
    def _divide_fmri_regions(fmri_data):
        """Divide 4D fMRI into 8 spatial regions"""
        if len(fmri_data.shape) == 4:
            x, y, z, t = fmri_data.shape
            x_mid, y_mid, z_mid = x // 2, y // 2, z // 2

            regions = [
                fmri_data[:x_mid, :y_mid, :z_mid, :],
                fmri_data[:x_mid, :y_mid, z_mid:, :],
                fmri_data[:x_mid, y_mid:, :z_mid, :],
                fmri_data[:x_mid, y_mid:, z_mid:, :],
                fmri_data[x_mid:, :y_mid, :z_mid, :],
                fmri_data[x_mid:, :y_mid, z_mid:, :],
                fmri_data[x_mid:, y_mid:, :z_mid, :],
                fmri_data[x_mid:, y_mid:, z_mid:, :]
            ]
        else:
            regions = []

        return regions

    @staticmethod
    def detect_cmri_bifurcations(cmri_data, method='all'):
        """
        Detect bifurcations in contrast-enhanced MRI

        cMRI shows contrast agent uptake, revealing:
        - Enhancement boundaries (tumor edges, blood-brain barrier breakdown)
        - Perfusion transitions
        - Vascular bifurcations

        Parameters:
        -----------
        cmri_data : ndarray
            Contrast-enhanced MRI data (2D, 3D, or 4D if dynamic)
        method : str
            'enhancement', 'perfusion', 'vascular', or 'all'

        Returns:
        --------
        bifurcations : dict
            cMRI bifurcation metrics
        """
        results = {'modality': 'cMRI'}

        # Determine dimensionality
        is_dynamic = len(cmri_data.shape) == 4

        if is_dynamic:
            # Dynamic contrast (4D: x, y, z, time)
            baseline = cmri_data[:, :, :, 0]
            peak_enhancement = np.max(cmri_data, axis=3)
        else:
            # Static contrast
            if len(cmri_data.shape) == 3:
                baseline = None
                peak_enhancement = cmri_data
            else:
                baseline = None
                peak_enhancement = cmri_data

        # Enhancement boundary bifurcations
        if method in ['enhancement', 'all']:
            # Detect sharp transitions in contrast enhancement
            if len(peak_enhancement.shape) == 3:
                middle_slice = peak_enhancement[:, :, peak_enhancement.shape[2] // 2]
            else:
                middle_slice = peak_enhancement

            enhancement_bif = NonLinearAnalyzer.detect_spatial_bifurcations(
                middle_slice, method='gradient'
            )

            results['enhancement_bifurcations'] = enhancement_bif['n_bifurcation_pixels']
            results['enhancement_bifurcation_density'] = enhancement_bif['bifurcation_density']
            results['enhancement_bifurcation_map'] = enhancement_bif['bifurcation_map']

            # High-intensity region boundaries (likely tumors)
            high_intensity_threshold = np.percentile(middle_slice, 90)
            high_regions = middle_slice > high_intensity_threshold

            # Detect boundaries of high-intensity regions
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(high_regions)
            boundaries = high_regions & ~eroded

            results['high_intensity_boundary_pixels'] = np.sum(boundaries)
            results['high_intensity_regions'] = np.sum(high_regions)

        # Perfusion bifurcations (if dynamic contrast available)
        if is_dynamic and method in ['perfusion', 'all']:
            # Detect temporal transitions in contrast uptake
            # Average across space
            global_enhancement = np.mean(cmri_data, axis=(0, 1, 2))

            # Time to peak
            time_to_peak = np.argmax(global_enhancement)

            # Detect perfusion transitions
            enhancement_curve_diff = np.abs(np.diff(global_enhancement))
            threshold = np.mean(enhancement_curve_diff) + 2 * np.std(enhancement_curve_diff)
            perfusion_transitions = np.where(enhancement_curve_diff > threshold)[0]

            results['perfusion_bifurcations'] = len(perfusion_transitions)
            results['perfusion_transition_times'] = perfusion_transitions
            results['time_to_peak'] = time_to_peak

            # Regional perfusion bifurcations
            regions = NonLinearAnalyzer._divide_fmri_regions(cmri_data)
            regional_perfusion_bif = []

            for region in regions:
                if region.size > 0:
                    region_curve = np.mean(region, axis=(0, 1, 2))
                    curve_diff = np.abs(np.diff(region_curve))
                    reg_threshold = np.mean(curve_diff) + 1.5 * np.std(curve_diff)
                    reg_transitions = np.sum(curve_diff > reg_threshold)
                    regional_perfusion_bif.append(reg_transitions)

            results['regional_perfusion_bifurcations'] = regional_perfusion_bif
            results['mean_regional_perfusion_bif'] = np.mean(regional_perfusion_bif)

        # Vascular bifurcations (vessel branching points)
        if method in ['vascular', 'all']:
            # Use Laplacian to detect vessel branching
            if len(peak_enhancement.shape) == 3:
                middle_slice = peak_enhancement[:, :, peak_enhancement.shape[2] // 2]
            else:
                middle_slice = peak_enhancement

            vascular_bif = NonLinearAnalyzer.detect_spatial_bifurcations(
                middle_slice, method='laplacian'
            )

            results['vascular_bifurcations'] = vascular_bif['n_bifurcation_pixels']
            results['vascular_bifurcation_density'] = vascular_bif['bifurcation_density']

        # Total bifurcations
        total = results.get('enhancement_bifurcations', 0)
        if 'perfusion_bifurcations' in results:
            total += results['perfusion_bifurcations']

        results['total_bifurcations'] = total

        return results

    @staticmethod
    def recurrence_based_bifurcation_detection(x, embedding_dim=3, delay=1):
        """
        Detect bifurcations using recurrence plot analysis

        Bifurcations show up as changes in recurrence structure

        Parameters:
        -----------
        x : array-like
            Time series
        embedding_dim : int
            Embedding dimension
        delay : int
            Time delay

        Returns:
        --------
        bifurcation_info : dict
            Recurrence-based bifurcation metrics
        """
        x = np.array(x)
        N = len(x)

        # Time-delay embedding
        M = N - (embedding_dim - 1) * delay
        if M < 10:
            return {'n_bifurcations': 0, 'determinism': 0}

        embedded = np.zeros((M, embedding_dim))
        for i in range(M):
            embedded[i] = x[i:i + embedding_dim * delay:delay]

        # Compute recurrence matrix
        distances = squareform(pdist(embedded, metric='euclidean'))
        threshold = 0.1 * np.max(distances)
        recurrence_matrix = (distances < threshold).astype(int)

        # Analyze diagonal lines (determinism)
        # Changes in determinism indicate bifurcations
        window = 50
        determinisms = []

        for i in range(0, M - window, window // 2):
            sub_matrix = recurrence_matrix[i:i+window, i:i+window]
            # Count diagonal lines
            diag_length = 0
            for d in range(-window+1, window):
                diag = np.diagonal(sub_matrix, d)
                diag_length += np.sum(diag)

            det = diag_length / (window * window)
            determinisms.append(det)

        determinisms = np.array(determinisms)

        # Detect sudden changes
        if len(determinisms) > 2:
            det_diff = np.abs(np.diff(determinisms))
            threshold_bif = np.mean(det_diff) + 1.5 * np.std(det_diff)
            bifurcations = np.where(det_diff > threshold_bif)[0]
        else:
            bifurcations = np.array([])

        return {
            'n_bifurcations': len(bifurcations),
            'bifurcation_indices': bifurcations,
            'determinism_trajectory': determinisms,
            'mean_determinism': np.mean(determinisms),
            'recurrence_matrix': recurrence_matrix
        }


def detect_bifurcations_by_modality(data, modality_type, **kwargs):
    """
    Unified bifurcation detection dispatcher for all modalities

    Parameters:
    -----------
    data : array-like or MNE object
        Data appropriate for the modality
    modality_type : str
        'EEG', 'MEG', 'MRI', 'fMRI', or 'cMRI'
    **kwargs : dict
        Additional parameters for specific methods

    Returns:
    --------
    bifurcations : dict
        Bifurcation metrics for the modality
    """
    if modality_type == 'EEG':
        # Extract signal if MNE object
        if hasattr(data, 'get_data'):
            signal = np.mean(data.get_data(), axis=0)
        else:
            signal = np.mean(data, axis=0) if len(data.shape) > 1 else data

        return NonLinearAnalyzer.detect_temporal_bifurcations(
            signal,
            window_size=kwargs.get('window_size', None),
            overlap=kwargs.get('overlap', 0.5)
        )

    elif modality_type == 'MEG':
        if hasattr(data, 'get_data'):
            meg_data = data.get_data()
        else:
            meg_data = data

        return NonLinearAnalyzer.detect_meg_bifurcations(
            meg_data,
            sfreq=kwargs.get('sfreq', 1000),
            window_size=kwargs.get('window_size', None),
            overlap=kwargs.get('overlap', 0.5)
        )

    elif modality_type == 'MRI':
        return NonLinearAnalyzer.detect_spatial_bifurcations(
            data,
            method=kwargs.get('method', 'gradient')
        )

    elif modality_type == 'fMRI':
        return NonLinearAnalyzer.detect_fmri_bifurcations(
            data,
            tr=kwargs.get('tr', 2.0),
            method=kwargs.get('method', 'spatio-temporal')
        )

    elif modality_type == 'cMRI':
        return NonLinearAnalyzer.detect_cmri_bifurcations(
            data,
            method=kwargs.get('method', 'all')
        )

    else:
        raise ValueError(f"Unknown modality type: {modality_type}")


def compare_bifurcations_cross_modal(bifurcation_results):
    """
    Compare bifurcations across multiple modalities (up to 5x5 matrix)

    Parameters:
    -----------
    bifurcation_results : dict
        Dictionary mapping modality names to their bifurcation results
        Example: {'EEG': eeg_bif, 'MRI': mri_bif, 'fMRI': fmri_bif}

    Returns:
    --------
    comparison : dict
        Cross-modal bifurcation comparison matrix and metrics
    """
    modalities = list(bifurcation_results.keys())
    n_modalities = len(modalities)

    if n_modalities < 2:
        return {'error': 'Need at least 2 modalities for comparison'}

    # Extract bifurcation counts for each modality
    bifurcation_counts = {}
    bifurcation_densities = {}

    for mod_name, mod_result in bifurcation_results.items():
        # Get total bifurcations
        if 'total_bifurcations' in mod_result:
            bifurcation_counts[mod_name] = mod_result['total_bifurcations']
        elif 'n_bifurcations' in mod_result:
            bifurcation_counts[mod_name] = mod_result['n_bifurcations']
        else:
            bifurcation_counts[mod_name] = 0

        # Get density if available
        if 'bifurcation_density' in mod_result:
            bifurcation_densities[mod_name] = mod_result['bifurcation_density']
        elif 'spatial_bifurcation_density' in mod_result:
            bifurcation_densities[mod_name] = mod_result['spatial_bifurcation_density']
        elif 'temporal_bifurcation_density' in mod_result:
            bifurcation_densities[mod_name] = mod_result['temporal_bifurcation_density']

    # Create correlation matrix
    corr_matrix = np.zeros((n_modalities, n_modalities))

    counts_array = np.array([bifurcation_counts[m] for m in modalities])

    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Normalized correlation based on bifurcation counts
                count1 = bifurcation_counts[mod1]
                count2 = bifurcation_counts[mod2]

                if count1 > 0 or count2 > 0:
                    max_count = max(count1, count2)
                    min_count = min(count1, count2)
                    corr = min_count / (max_count + 1e-10)
                else:
                    corr = 0

                corr_matrix[i, j] = corr

    # Find dominant modality
    max_bifurcations = max(bifurcation_counts.values())
    dominant_modality = [m for m, c in bifurcation_counts.items() if c == max_bifurcations][0]

    # Complexity balance
    complexity_variance = np.var(counts_array)
    mean_complexity = np.mean(counts_array)

    if mean_complexity > 0:
        complexity_cv = complexity_variance / mean_complexity
    else:
        complexity_cv = 0

    # Interpretation
    if complexity_cv < 0.3:
        interpretation = "Balanced: All modalities show similar bifurcation complexity"
    elif complexity_cv < 0.7:
        interpretation = f"Moderate imbalance: {dominant_modality} shows more bifurcations"
    else:
        interpretation = f"High imbalance: {dominant_modality} dominant, others stable"

    return {
        'modalities': modalities,
        'bifurcation_counts': bifurcation_counts,
        'bifurcation_densities': bifurcation_densities,
        'correlation_matrix': corr_matrix,
        'dominant_modality': dominant_modality,
        'complexity_variance': complexity_variance,
        'complexity_coefficient_of_variation': complexity_cv,
        'interpretation': interpretation,
        'mean_bifurcations': mean_complexity,
        'total_bifurcations_all_modalities': np.sum(counts_array)
    }


class TensorAnalyzer:
    """
    Tensor-Based Analysis for Multi-Dimensional Neuroimaging Data

    Preserves natural tensor structure instead of flattening to matrices.
    Much more efficient and preserves spatial/temporal/spectral relationships.
    """

    @staticmethod
    def tensor_psd(tensor_data, sfreq, axis=-1, method='welch', nperseg=None):
        """
        Power spectral density preserving tensor structure

        Instead of flattening, compute PSD along time axis while preserving
        spatial dimensions.

        Parameters:
        -----------
        tensor_data : ndarray
            Multi-dimensional data (e.g., channels x time, or x,y,z,time)
        sfreq : float
            Sampling frequency
        axis : int
            Time axis (default: -1, last axis)
        method : str
            'welch' or 'periodogram'
        nperseg : int
            Segment length for Welch

        Returns:
        --------
        result : dict
            'freqs': Frequency bins
            'psd_tensor': PSD with same shape as input (time → freq)
            'band_power_tensors': Band powers preserving spatial structure
        """
        if nperseg is None:
            nperseg = min(256, tensor_data.shape[axis] // 4)

        # Move time axis to last position
        tensor_moved = np.moveaxis(tensor_data, axis, -1)
        original_shape = tensor_moved.shape[:-1]

        # Reshape to (n_positions, n_timepoints)
        tensor_2d = tensor_moved.reshape(-1, tensor_moved.shape[-1])

        # Compute PSD for all positions at once
        if method == 'welch':
            freqs, psd_2d = signal.welch(tensor_2d, fs=sfreq, nperseg=nperseg, axis=-1)
        elif method == 'periodogram':
            freqs, psd_2d = signal.periodogram(tensor_2d, fs=sfreq, axis=-1)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Reshape back to original spatial structure + freqs
        psd_tensor = psd_2d.reshape(*original_shape, len(freqs))

        # Calculate band powers preserving spatial structure
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        band_power_tensors = {}
        for band_name, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            if np.any(idx):
                # Integrate over frequency
                band_power_tensors[band_name] = np.trapz(
                    psd_tensor[..., idx], freqs[idx], axis=-1
                )
            else:
                band_power_tensors[band_name] = np.zeros(original_shape)

        return {
            'freqs': freqs,
            'psd_tensor': psd_tensor,
            'band_power_tensors': band_power_tensors,
            'original_shape': tensor_data.shape,
            'spatial_shape': original_shape
        }

    @staticmethod
    def tensor_coherence(tensor1, tensor2, sfreq, axis=-1, nperseg=256):
        """
        Coherence between two tensors preserving spatial structure

        Parameters:
        -----------
        tensor1, tensor2 : ndarray
            Multi-dimensional tensors (must have same shape)
        sfreq : float
            Sampling frequency
        axis : int
            Time axis
        nperseg : int
            Segment length

        Returns:
        --------
        result : dict
            'freqs': Frequency bins
            'coherence_tensor': Coherence preserving spatial structure
            'mean_coherence_tensor': Mean coherence across frequencies
        """
        assert tensor1.shape == tensor2.shape, "Tensors must have same shape"

        # Move time axis to last position
        t1_moved = np.moveaxis(tensor1, axis, -1)
        t2_moved = np.moveaxis(tensor2, axis, -1)

        original_shape = t1_moved.shape[:-1]

        # Reshape to (n_positions, n_timepoints)
        t1_2d = t1_moved.reshape(-1, t1_moved.shape[-1])
        t2_2d = t2_moved.reshape(-1, t2_moved.shape[-1])

        # Compute coherence for all positions
        coherence_list = []
        freqs = None

        for i in range(t1_2d.shape[0]):
            f, coh = signal.coherence(t1_2d[i], t2_2d[i], fs=sfreq, nperseg=nperseg)
            coherence_list.append(coh)
            if freqs is None:
                freqs = f

        coherence_2d = np.array(coherence_list)

        # Reshape back to tensor
        coherence_tensor = coherence_2d.reshape(*original_shape, len(freqs))

        # Mean coherence across frequencies
        mean_coherence_tensor = np.mean(coherence_tensor, axis=-1)

        return {
            'freqs': freqs,
            'coherence_tensor': coherence_tensor,
            'mean_coherence_tensor': mean_coherence_tensor,
            'spatial_shape': original_shape
        }

    @staticmethod
    def tensor_cp_decomposition(tensor_data, rank=3, max_iter=100):
        """
        CP/PARAFAC tensor decomposition

        Decomposes N-way tensor into sum of rank-1 tensors:
        X ≈ Σ λᵣ (a_r ⊗ b_r ⊗ c_r ⊗ ...)

        Parameters:
        -----------
        tensor_data : ndarray
            N-dimensional tensor
        rank : int
            Number of components
        max_iter : int
            Maximum iterations

        Returns:
        --------
        result : dict
            'factors': List of factor matrices (one per mode)
            'weights': Component weights
            'reconstruction': Reconstructed tensor
            'error': Reconstruction error
        """
        try:
            import tensorly as tl
            from tensorly.decomposition import parafac

            # Convert to tensorly tensor
            tl_tensor = tl.tensor(tensor_data)

            # CP decomposition
            cp_tensor = parafac(tl_tensor, rank=rank, n_iter_max=max_iter)

            # Extract factors and weights
            weights, factors = cp_tensor

            # Reconstruct
            reconstruction = tl.cp_to_tensor(cp_tensor)

            # Error
            error = np.linalg.norm(tensor_data - reconstruction) / np.linalg.norm(tensor_data)

            return {
                'factors': factors,
                'weights': weights,
                'reconstruction': reconstruction,
                'error': error,
                'rank': rank,
                'n_modes': len(factors)
            }

        except ImportError:
            # Fallback: Simple alternating least squares
            return TensorAnalyzer._simple_cp_als(tensor_data, rank, max_iter)

    @staticmethod
    def _simple_cp_als(tensor_data, rank, max_iter):
        """Simple CP-ALS implementation without tensorly"""
        ndim = tensor_data.ndim
        shape = tensor_data.shape

        # Initialize factor matrices randomly
        factors = [np.random.rand(shape[i], rank) for i in range(ndim)]

        for iteration in range(max_iter):
            for mode in range(ndim):
                # Matricize tensor along mode
                unfolding = TensorAnalyzer._unfold_tensor(tensor_data, mode)

                # Compute Khatri-Rao product of all factors except mode
                kr_product = TensorAnalyzer._khatri_rao([factors[i] for i in range(ndim) if i != mode])

                # Update factor
                factors[mode] = unfolding @ kr_product @ np.linalg.pinv(kr_product.T @ kr_product)

        # Weights (norms of factors)
        weights = np.array([np.linalg.norm(factors[0][:, r]) for r in range(rank)])

        # Normalize factors
        for mode in range(ndim):
            for r in range(rank):
                norm = np.linalg.norm(factors[mode][:, r])
                if norm > 0:
                    factors[mode][:, r] /= norm

        # Reconstruct
        reconstruction = TensorAnalyzer._cp_reconstruct(factors, weights)

        # Error
        error = np.linalg.norm(tensor_data - reconstruction) / np.linalg.norm(tensor_data)

        return {
            'factors': factors,
            'weights': weights,
            'reconstruction': reconstruction,
            'error': error,
            'rank': rank,
            'n_modes': ndim
        }

    @staticmethod
    def tensor_tucker_decomposition(tensor_data, ranks=None, max_iter=100):
        """
        Tucker tensor decomposition

        Decomposes tensor as: X ≈ G ×₁ A ×₂ B ×₃ C ...

        Parameters:
        -----------
        tensor_data : ndarray
            N-dimensional tensor
        ranks : list of int
            Rank for each mode (default: half of each dimension)
        max_iter : int
            Maximum iterations

        Returns:
        --------
        result : dict
            'core': Core tensor
            'factors': List of factor matrices
            'reconstruction': Reconstructed tensor
            'error': Reconstruction error
        """
        try:
            import tensorly as tl
            from tensorly.decomposition import tucker

            if ranks is None:
                ranks = [max(1, s // 2) for s in tensor_data.shape]

            # Convert to tensorly tensor
            tl_tensor = tl.tensor(tensor_data)

            # Tucker decomposition
            tucker_tensor = tucker(tl_tensor, rank=ranks, n_iter_max=max_iter)

            # Extract core and factors
            core, factors = tucker_tensor

            # Reconstruct
            reconstruction = tl.tucker_to_tensor(tucker_tensor)

            # Error
            error = np.linalg.norm(tensor_data - reconstruction) / np.linalg.norm(tensor_data)

            return {
                'core': core,
                'factors': factors,
                'reconstruction': reconstruction,
                'error': error,
                'ranks': ranks,
                'compression_ratio': np.prod(tensor_data.shape) / (
                    np.prod(core.shape) + sum(f.size for f in factors)
                )
            }

        except ImportError:
            # Fallback: Higher-Order SVD (HOSVD)
            return TensorAnalyzer._hosvd(tensor_data, ranks)

    @staticmethod
    def _hosvd(tensor_data, ranks):
        """Higher-Order SVD (deterministic Tucker approximation)"""
        if ranks is None:
            ranks = [max(1, s // 2) for s in tensor_data.shape]

        ndim = tensor_data.ndim
        factors = []

        # Compute factor matrices via SVD of unfoldings
        for mode in range(ndim):
            unfolding = TensorAnalyzer._unfold_tensor(tensor_data, mode)
            U, S, Vt = np.linalg.svd(unfolding, full_matrices=False)
            factors.append(U[:, :ranks[mode]])

        # Compute core tensor
        core = tensor_data.copy()
        for mode in range(ndim):
            core = TensorAnalyzer._mode_n_product(core, factors[mode].T, mode)

        # Reconstruct
        reconstruction = core.copy()
        for mode in range(ndim):
            reconstruction = TensorAnalyzer._mode_n_product(reconstruction, factors[mode], mode)

        # Error
        error = np.linalg.norm(tensor_data - reconstruction) / np.linalg.norm(tensor_data)

        return {
            'core': core,
            'factors': factors,
            'reconstruction': reconstruction,
            'error': error,
            'ranks': ranks,
            'compression_ratio': np.prod(tensor_data.shape) / (
                np.prod(core.shape) + sum(f.size for f in factors)
            )
        }

    @staticmethod
    def _unfold_tensor(tensor_data, mode):
        """Unfold tensor along specified mode (matricization)"""
        shape = tensor_data.shape
        new_shape = (shape[mode], -1)
        # Move mode to front, then reshape
        axes = [mode] + [i for i in range(len(shape)) if i != mode]
        return np.moveaxis(tensor_data, axes, range(len(axes))).reshape(new_shape)

    @staticmethod
    def _mode_n_product(tensor_data, matrix, mode):
        """Mode-n product: tensor ×_n matrix"""
        shape = tensor_data.shape
        # Unfold tensor along mode
        unfolding = TensorAnalyzer._unfold_tensor(tensor_data, mode)
        # Matrix multiply
        result = matrix @ unfolding
        # Fold back
        new_shape = list(shape)
        new_shape[mode] = matrix.shape[0]
        # Reshape and move axis back
        result_tensor = result.reshape([new_shape[mode]] + [new_shape[i] for i in range(len(shape)) if i != mode])
        axes = list(range(1, mode + 1)) + [0] + list(range(mode + 1, len(shape)))
        return np.moveaxis(result_tensor, range(len(axes)), axes)

    @staticmethod
    def _khatri_rao(matrices):
        """Khatri-Rao product (column-wise Kronecker)"""
        if len(matrices) == 1:
            return matrices[0]

        result = matrices[0]
        for mat in matrices[1:]:
            n_cols = result.shape[1]
            kr = np.zeros((result.shape[0] * mat.shape[0], n_cols))
            for col in range(n_cols):
                kr[:, col] = np.kron(result[:, col], mat[:, col])
            result = kr

        return result

    @staticmethod
    def _cp_reconstruct(factors, weights):
        """Reconstruct tensor from CP factors"""
        rank = len(weights)
        ndim = len(factors)

        # Initialize reconstruction
        shape = tuple(f.shape[0] for f in factors)
        reconstruction = np.zeros(shape)

        # Sum rank-1 tensors
        for r in range(rank):
            # Outer product of all factors for component r
            rank1_tensor = factors[0][:, r]
            for mode in range(1, ndim):
                rank1_tensor = np.multiply.outer(rank1_tensor, factors[mode][:, r])

            reconstruction += weights[r] * rank1_tensor

        return reconstruction

    @staticmethod
    def tensor_bifurcation_detection(tensor_data, axis=-1, method='gradient', threshold=None):
        """
        Detect bifurcations in tensor data preserving spatial structure

        Parameters:
        -----------
        tensor_data : ndarray
            Multi-dimensional tensor
        axis : int
            Temporal axis along which to detect bifurcations
        method : str
            'gradient', 'variance', or 'spectral'
        threshold : float
            Detection threshold (auto if None)

        Returns:
        --------
        result : dict
            'bifurcation_tensor': Binary tensor marking bifurcations
            'bifurcation_strength': Strength at each position
            'bifurcation_times': Time indices where bifurcations occur
        """
        # Move time axis to last position
        tensor_moved = np.moveaxis(tensor_data, axis, -1)
        spatial_shape = tensor_moved.shape[:-1]
        n_timepoints = tensor_moved.shape[-1]

        if method == 'gradient':
            # Temporal gradient (preserving spatial structure)
            gradient = np.abs(np.diff(tensor_moved, axis=-1))

            # Detection threshold
            if threshold is None:
                threshold = np.mean(gradient) + 2 * np.std(gradient)

            # Bifurcations
            bifurcation_binary = gradient > threshold

            # Strength
            bifurcation_strength = gradient

        elif method == 'variance':
            # Sliding window variance
            window_size = min(10, n_timepoints // 10)
            variances = np.zeros_like(tensor_moved)

            for t in range(n_timepoints - window_size):
                window = tensor_moved[..., t:t+window_size]
                variances[..., t] = np.var(window, axis=-1)

            # Detection
            if threshold is None:
                threshold = np.mean(variances) + 2 * np.std(variances)

            bifurcation_binary = variances > threshold
            bifurcation_strength = variances

        elif method == 'spectral':
            # Spectral bifurcations preserving tensor structure
            tensor_2d = tensor_moved.reshape(-1, n_timepoints)
            spectral_changes = np.zeros((tensor_2d.shape[0], n_timepoints - 1))

            window_size = min(256, n_timepoints // 4)
            for i in range(tensor_2d.shape[0]):
                for t in range(0, n_timepoints - window_size - 1, window_size // 2):
                    w1 = tensor_2d[i, t:t+window_size]
                    w2 = tensor_2d[i, t+window_size//2:t+3*window_size//2]

                    if len(w2) >= window_size:
                        fft1 = np.abs(np.fft.rfft(w1))
                        fft2 = np.abs(np.fft.rfft(w2))

                        peak1 = np.argmax(fft1)
                        peak2 = np.argmax(fft2)

                        change = abs(peak1 - peak2)
                        t_idx = t + window_size // 2
                        if t_idx < spectral_changes.shape[1]:
                            spectral_changes[i, t_idx] = change

            # Reshape back
            bifurcation_strength = spectral_changes.reshape(*spatial_shape, -1)

            if threshold is None:
                threshold = np.mean(spectral_changes) + 2 * np.std(spectral_changes)

            bifurcation_binary = bifurcation_strength > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        # Find time indices with bifurcations (anywhere in spatial dims)
        bifurcation_per_time = np.any(bifurcation_binary.reshape(-1, bifurcation_binary.shape[-1]), axis=0)
        bifurcation_times = np.where(bifurcation_per_time)[0]

        return {
            'bifurcation_tensor': bifurcation_binary,
            'bifurcation_strength': bifurcation_strength,
            'bifurcation_times': bifurcation_times,
            'n_bifurcations': len(bifurcation_times),
            'spatial_shape': spatial_shape,
            'method': method
        }

    @staticmethod
    def tensor_einsum_correlation(tensor1, tensor2, mode='spatial'):
        """
        Efficient tensor correlation using einsum

        Parameters:
        -----------
        tensor1, tensor2 : ndarray
            Tensors to correlate
        mode : str
            'spatial': Correlate spatial patterns across time
            'temporal': Correlate temporal patterns across space
            'full': Full tensor correlation

        Returns:
        --------
        correlation : float or ndarray
            Correlation value(s)
        """
        # Normalize tensors
        t1_normalized = (tensor1 - np.mean(tensor1)) / (np.std(tensor1) + 1e-10)
        t2_normalized = (tensor2 - np.mean(tensor2)) / (np.std(tensor2) + 1e-10)

        if mode == 'full':
            # Full tensor correlation (single value)
            correlation = np.sum(t1_normalized * t2_normalized) / tensor1.size

        elif mode == 'spatial':
            # For each time point, correlate spatial patterns
            # Assumes last axis is time
            n_time = tensor1.shape[-1]
            t1_spatial = t1_normalized.reshape(-1, n_time)
            t2_spatial = t2_normalized.reshape(-1, n_time)

            # Correlation at each time point
            correlation = np.array([
                np.corrcoef(t1_spatial[:, t], t2_spatial[:, t])[0, 1]
                for t in range(n_time)
            ])

        elif mode == 'temporal':
            # For each spatial location, correlate time series
            # Assumes last axis is time
            n_spatial = np.prod(tensor1.shape[:-1])
            t1_temporal = t1_normalized.reshape(n_spatial, -1)
            t2_temporal = t2_normalized.reshape(n_spatial, -1)

            # Correlation at each spatial location
            correlation = np.array([
                np.corrcoef(t1_temporal[s], t2_temporal[s])[0, 1]
                for s in range(n_spatial)
            ])

            # Reshape back to spatial dimensions
            correlation = correlation.reshape(tensor1.shape[:-1])

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return correlation


class TransformAnalyzer:
    """
    Transform-Based Analysis for Neuroimaging Data

    Uses Fourier, Wavelet, and other transforms to simplify calculations
    and extract frequency-domain features essential for neuroimaging.
    """

    @staticmethod
    def power_spectral_density(signal_data, sfreq, method='welch', nperseg=None):
        """
        Calculate power spectral density using FFT-based methods

        Parameters:
        -----------
        signal_data : array-like
            Time series data
        sfreq : float
            Sampling frequency
        method : str
            'welch' (recommended), 'periodogram', or 'multitaper'
        nperseg : int
            Segment length for Welch method (default: 256)

        Returns:
        --------
        result : dict
            'freqs': Frequency bins
            'psd': Power spectral density
            'total_power': Total power
            'band_powers': Power in standard frequency bands
        """
        if nperseg is None:
            nperseg = min(256, len(signal_data) // 4)

        if method == 'welch':
            freqs, psd = signal.welch(signal_data, fs=sfreq, nperseg=nperseg)
        elif method == 'periodogram':
            freqs, psd = signal.periodogram(signal_data, fs=sfreq)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate band powers (for EEG/MEG)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            if np.any(idx):
                band_powers[band_name] = np.trapz(psd[idx], freqs[idx])
            else:
                band_powers[band_name] = 0

        total_power = np.trapz(psd, freqs)

        return {
            'freqs': freqs,
            'psd': psd,
            'total_power': total_power,
            'band_powers': band_powers,
            'dominant_freq': freqs[np.argmax(psd)],
            'peak_power': np.max(psd)
        }

    @staticmethod
    def wavelet_transform(signal_data, sfreq, wavelet='morl', scales=None):
        """
        Continuous Wavelet Transform (CWT) for time-frequency analysis

        Parameters:
        -----------
        signal_data : array-like
            Time series data
        sfreq : float
            Sampling frequency
        wavelet : str
            Wavelet name ('morl', 'cmor', 'mexh', etc.)
        scales : array-like
            Scales to use (default: auto-generate)

        Returns:
        --------
        result : dict
            'coefficients': Wavelet coefficients (scales x time)
            'freqs': Corresponding frequencies
            'scales': Scales used
            'power': Time-frequency power |coef|²
        """
        import pywt

        if scales is None:
            # Auto-generate scales for 0.5 Hz to Nyquist
            freqs_desired = np.linspace(0.5, sfreq / 2, 50)
            scales = pywt.frequency2scale(wavelet, freqs_desired) / sfreq

        # Compute CWT
        coefficients, freqs = pywt.cwt(
            signal_data,
            scales,
            wavelet,
            sampling_period=1/sfreq
        )

        # Time-frequency power
        power = np.abs(coefficients) ** 2

        return {
            'coefficients': coefficients,
            'freqs': freqs,
            'scales': scales,
            'power': power,
            'time': np.arange(len(signal_data)) / sfreq
        }

    @staticmethod
    def short_time_fourier_transform(signal_data, sfreq, nperseg=256, noverlap=None):
        """
        Short-Time Fourier Transform (STFT) for spectrogram

        Parameters:
        -----------
        signal_data : array-like
            Time series data
        sfreq : float
            Sampling frequency
        nperseg : int
            Segment length
        noverlap : int
            Overlap length (default: nperseg // 2)

        Returns:
        --------
        result : dict
            'freqs': Frequency bins
            'times': Time bins
            'spectrogram': STFT magnitude (freqs x times)
            'phase': STFT phase
        """
        if noverlap is None:
            noverlap = nperseg // 2

        freqs, times, Zxx = signal.stft(
            signal_data,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        return {
            'freqs': freqs,
            'times': times,
            'spectrogram': magnitude,
            'phase': phase,
            'complex': Zxx
        }

    @staticmethod
    def coherence_analysis(signal1, signal2, sfreq, nperseg=256):
        """
        Magnitude-squared coherence between two signals (FFT-based)

        Coherence measures linear correlation in frequency domain

        Parameters:
        -----------
        signal1, signal2 : array-like
            Time series data
        sfreq : float
            Sampling frequency
        nperseg : int
            Segment length

        Returns:
        --------
        result : dict
            'freqs': Frequency bins
            'coherence': Coherence values [0, 1]
            'mean_coherence': Average across frequencies
            'band_coherence': Coherence per frequency band
        """
        freqs, coh = signal.coherence(signal1, signal2, fs=sfreq, nperseg=nperseg)

        # Band-specific coherence
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        band_coherence = {}
        for band_name, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            if np.any(idx):
                band_coherence[band_name] = np.mean(coh[idx])
            else:
                band_coherence[band_name] = 0

        return {
            'freqs': freqs,
            'coherence': coh,
            'mean_coherence': np.mean(coh),
            'band_coherence': band_coherence
        }

    @staticmethod
    def cross_spectral_density(signal1, signal2, sfreq, nperseg=256):
        """
        Cross-spectral density between two signals

        Parameters:
        -----------
        signal1, signal2 : array-like
            Time series data
        sfreq : float
            Sampling frequency
        nperseg : int
            Segment length

        Returns:
        --------
        result : dict
            'freqs': Frequency bins
            'csd': Cross-spectral density (complex)
            'magnitude': |CSD|
            'phase': Phase difference
        """
        freqs, csd = signal.csd(signal1, signal2, fs=sfreq, nperseg=nperseg)

        magnitude = np.abs(csd)
        phase = np.angle(csd)

        return {
            'freqs': freqs,
            'csd': csd,
            'magnitude': magnitude,
            'phase': phase
        }

    @staticmethod
    def frequency_domain_bifurcation_detection(signal_data, sfreq, method='spectral_edge'):
        """
        Detect bifurcations using frequency-domain features

        Bifurcations often manifest as spectral changes (peak shifts,
        bandwidth changes, power redistribution)

        Parameters:
        -----------
        signal_data : array-like
            Time series data
        sfreq : float
            Sampling frequency
        method : str
            'spectral_edge', 'peak_shift', 'bandwidth_change'

        Returns:
        --------
        result : dict
            'bifurcation_indices': Time points of bifurcations
            'spectral_features': Time-varying spectral features
            'bifurcation_scores': Bifurcation strength
        """
        # Use STFT to get time-varying spectrum
        nperseg = min(256, len(signal_data) // 10)
        stft_result = TransformAnalyzer.short_time_fourier_transform(
            signal_data, sfreq, nperseg=nperseg
        )

        times = stft_result['times']
        spectrogram = stft_result['spectrogram']
        freqs = stft_result['freqs']

        if method == 'spectral_edge':
            # Spectral edge frequency (95% power threshold)
            spectral_edges = []
            for t_idx in range(spectrogram.shape[1]):
                psd = spectrogram[:, t_idx]
                cumsum = np.cumsum(psd)
                total = cumsum[-1]
                if total > 0:
                    threshold_idx = np.where(cumsum >= 0.95 * total)[0]
                    if len(threshold_idx) > 0:
                        spectral_edges.append(freqs[threshold_idx[0]])
                    else:
                        spectral_edges.append(freqs[-1])
                else:
                    spectral_edges.append(0)

            spectral_edges = np.array(spectral_edges)

            # Bifurcations = large changes in spectral edge
            edge_diff = np.abs(np.diff(spectral_edges))
            threshold = np.mean(edge_diff) + 2 * np.std(edge_diff)
            bifurcation_indices = np.where(edge_diff > threshold)[0]
            bifurcation_scores = edge_diff[bifurcation_indices]

        elif method == 'peak_shift':
            # Dominant frequency shifts
            peak_freqs = []
            for t_idx in range(spectrogram.shape[1]):
                psd = spectrogram[:, t_idx]
                peak_freqs.append(freqs[np.argmax(psd)])

            peak_freqs = np.array(peak_freqs)

            # Bifurcations = sudden peak shifts
            peak_diff = np.abs(np.diff(peak_freqs))
            threshold = np.mean(peak_diff) + 2 * np.std(peak_diff)
            bifurcation_indices = np.where(peak_diff > threshold)[0]
            bifurcation_scores = peak_diff[bifurcation_indices]

        elif method == 'bandwidth_change':
            # Spectral bandwidth changes
            bandwidths = []
            for t_idx in range(spectrogram.shape[1]):
                psd = spectrogram[:, t_idx]
                if psd.sum() > 0:
                    psd_norm = psd / psd.sum()
                    mean_freq = np.sum(freqs * psd_norm)
                    variance = np.sum(((freqs - mean_freq) ** 2) * psd_norm)
                    bandwidth = np.sqrt(variance)
                    bandwidths.append(bandwidth)
                else:
                    bandwidths.append(0)

            bandwidths = np.array(bandwidths)

            # Bifurcations = large bandwidth changes
            bw_diff = np.abs(np.diff(bandwidths))
            threshold = np.mean(bw_diff) + 2 * np.std(bw_diff)
            bifurcation_indices = np.where(bw_diff > threshold)[0]
            bifurcation_scores = bw_diff[bifurcation_indices]

        else:
            raise ValueError(f"Unknown method: {method}")

        # Map back to original time indices
        time_per_segment = times[1] - times[0] if len(times) > 1 else 1
        original_indices = (bifurcation_indices * sfreq * time_per_segment).astype(int)

        return {
            'bifurcation_indices': original_indices,
            'bifurcation_times': times[bifurcation_indices] if len(bifurcation_indices) > 0 else np.array([]),
            'bifurcation_scores': bifurcation_scores,
            'spectral_features': {
                'times': times,
                'spectrogram': spectrogram,
                'freqs': freqs
            },
            'n_bifurcations': len(bifurcation_indices)
        }

    @staticmethod
    def hilbert_huang_transform(signal_data, sfreq, n_imfs=5):
        """
        Hilbert-Huang Transform (EMD + Hilbert)

        Decomposes signal into Intrinsic Mode Functions (IMFs) and
        calculates instantaneous frequency/amplitude

        Parameters:
        -----------
        signal_data : array-like
            Time series data
        sfreq : float
            Sampling frequency
        n_imfs : int
            Maximum number of IMFs to extract

        Returns:
        --------
        result : dict
            'imfs': Intrinsic Mode Functions
            'instantaneous_freqs': Instantaneous frequencies per IMF
            'instantaneous_amps': Instantaneous amplitudes per IMF
            'hilbert_spectrum': Time-frequency representation
        """
        try:
            from PyEMD import EMD
        except ImportError:
            # Fallback: Simple implementation
            return TransformAnalyzer._simple_emd(signal_data, sfreq, n_imfs)

        # Empirical Mode Decomposition
        emd = EMD()
        imfs = emd(signal_data, max_imf=n_imfs)

        # Hilbert transform of each IMF
        instantaneous_freqs = []
        instantaneous_amps = []

        for imf in imfs:
            analytic_signal = signal.hilbert(imf)
            amplitude = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
            inst_freq = np.diff(phase) / (2.0 * np.pi) * sfreq
            inst_freq = np.append(inst_freq, inst_freq[-1])  # Maintain length

            instantaneous_amps.append(amplitude)
            instantaneous_freqs.append(inst_freq)

        return {
            'imfs': imfs,
            'instantaneous_freqs': np.array(instantaneous_freqs),
            'instantaneous_amps': np.array(instantaneous_amps),
            'time': np.arange(len(signal_data)) / sfreq,
            'n_imfs': len(imfs)
        }

    @staticmethod
    def _simple_emd(signal_data, sfreq, n_imfs):
        """Simple EMD fallback if PyEMD not available"""
        # Basic implementation using filtering
        imfs = []
        residue = signal_data.copy()

        for i in range(n_imfs):
            if len(residue) < 10:
                break

            # Use bandpass filter as simple IMF extraction
            nyq = sfreq / 2
            low = max(0.5, nyq / (2 ** (i + 2)))
            high = min(nyq - 1, nyq / (2 ** (i + 1)))

            if low >= high:
                break

            try:
                sos = signal.butter(4, [low, high], btype='band', fs=sfreq, output='sos')
                imf = signal.sosfilt(sos, residue)
                imfs.append(imf)
                residue = residue - imf
            except:
                break

        imfs = np.array(imfs) if imfs else np.array([signal_data])

        # Hilbert analysis
        instantaneous_freqs = []
        instantaneous_amps = []

        for imf in imfs:
            analytic_signal = signal.hilbert(imf)
            amplitude = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
            inst_freq = np.diff(phase) / (2.0 * np.pi) * sfreq
            inst_freq = np.append(inst_freq, inst_freq[-1])

            instantaneous_amps.append(amplitude)
            instantaneous_freqs.append(inst_freq)

        return {
            'imfs': imfs,
            'instantaneous_freqs': np.array(instantaneous_freqs),
            'instantaneous_amps': np.array(instantaneous_amps),
            'time': np.arange(len(signal_data)) / sfreq,
            'n_imfs': len(imfs)
        }

    @staticmethod
    def fft_based_correlation(signal1, signal2):
        """
        Fast correlation using FFT (O(n log n) instead of O(n²))

        Parameters:
        -----------
        signal1, signal2 : array-like
            Signals to correlate

        Returns:
        --------
        correlation : array-like
            Cross-correlation
        """
        # Ensure same length
        n = len(signal1)
        m = len(signal2)
        length = max(n, m)

        # Zero-pad to next power of 2 for efficiency
        fft_len = 2 ** int(np.ceil(np.log2(2 * length - 1)))

        # FFT-based correlation
        fft1 = np.fft.fft(signal1, n=fft_len)
        fft2 = np.fft.fft(signal2, n=fft_len)

        # Cross-correlation in frequency domain
        correlation = np.fft.ifft(fft1 * np.conj(fft2)).real

        # Return relevant portion
        correlation = correlation[:length]

        return correlation


class FuzzySetAnalyzer:
    """
    Fuzzy Set Theory Analysis for Neuroimaging Data

    Handles uncertainty and vagueness in bifurcation detection, state classification,
    and cross-modal comparisons using fuzzy logic.
    """

    @staticmethod
    def gaussian_membership(x, center, sigma):
        """
        Gaussian fuzzy membership function

        Parameters:
        -----------
        x : array-like
            Input values
        center : float
            Center of the Gaussian
        sigma : float
            Standard deviation (spread)

        Returns:
        --------
        membership : array-like
            Membership values [0, 1]
        """
        return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def triangular_membership(x, a, b, c):
        """
        Triangular fuzzy membership function

        Parameters:
        -----------
        x : array-like
            Input values
        a, b, c : float
            Left, center, and right points of triangle

        Returns:
        --------
        membership : array-like
            Membership values [0, 1]
        """
        membership = np.zeros_like(x, dtype=float)

        # Left slope
        left_mask = (x >= a) & (x < b)
        if b != a:
            membership[left_mask] = (x[left_mask] - a) / (b - a)

        # Right slope
        right_mask = (x >= b) & (x <= c)
        if c != b:
            membership[right_mask] = (c - x[right_mask]) / (c - b)

        return membership

    @staticmethod
    def trapezoidal_membership(x, a, b, c, d):
        """
        Trapezoidal fuzzy membership function

        Parameters:
        -----------
        x : array-like
            Input values
        a, b, c, d : float
            Left start, left top, right top, right end

        Returns:
        --------
        membership : array-like
            Membership values [0, 1]
        """
        membership = np.zeros_like(x, dtype=float)

        # Left slope
        left_mask = (x >= a) & (x < b)
        if b != a:
            membership[left_mask] = (x[left_mask] - a) / (b - a)

        # Top plateau
        top_mask = (x >= b) & (x <= c)
        membership[top_mask] = 1.0

        # Right slope
        right_mask = (x > c) & (x <= d)
        if d != c:
            membership[right_mask] = (d - x[right_mask]) / (d - c)

        return membership

    @staticmethod
    def fuzzy_cmeans_clustering(data, n_clusters=3, m=2, max_iter=100, error=1e-5):
        """
        Fuzzy C-Means clustering for pattern detection

        Parameters:
        -----------
        data : ndarray
            Data to cluster (n_samples, n_features)
        n_clusters : int
            Number of clusters
        m : float
            Fuzziness parameter (m > 1, typically 2)
        max_iter : int
            Maximum iterations
        error : float
            Convergence threshold

        Returns:
        --------
        result : dict
            'centers': Cluster centers
            'membership': Fuzzy membership matrix (n_samples, n_clusters)
            'labels': Hard labels (highest membership)
        """
        n_samples = data.shape[0]

        # Initialize random membership matrix
        membership = np.random.rand(n_samples, n_clusters)
        membership = membership / membership.sum(axis=1, keepdims=True)

        for iteration in range(max_iter):
            # Save old membership for convergence check
            old_membership = membership.copy()

            # Calculate cluster centers
            um = membership ** m
            centers = (um.T @ data) / um.sum(axis=0, keepdims=True).T

            # Update membership values
            for i in range(n_samples):
                for j in range(n_clusters):
                    distances = np.linalg.norm(data[i] - centers, axis=1)
                    distances[distances == 0] = 1e-10  # Avoid division by zero

                    # Fuzzy membership calculation
                    membership[i, j] = 1.0 / np.sum(
                        (distances[j] / distances) ** (2 / (m - 1))
                    )

            # Check convergence
            if np.linalg.norm(membership - old_membership) < error:
                break

        # Hard labels (highest membership)
        labels = np.argmax(membership, axis=1)

        return {
            'centers': centers,
            'membership': membership,
            'labels': labels,
            'n_iterations': iteration + 1
        }

    @staticmethod
    def fuzzy_bifurcation_detection(signal, threshold_range=(0.1, 0.5),
                                    membership_type='gaussian'):
        """
        Detect bifurcations with fuzzy boundaries (handling uncertainty)

        Parameters:
        -----------
        signal : array-like
            Input signal (1D time series or flattened spatial data)
        threshold_range : tuple
            (low, high) thresholds for fuzzy membership
        membership_type : str
            'gaussian', 'triangular', or 'trapezoidal'

        Returns:
        --------
        result : dict
            'crisp_bifurcations': Traditional hard threshold bifurcations
            'fuzzy_bifurcations': Bifurcations with membership degrees
            'fuzzy_states': State membership over time
            'uncertainty_map': Uncertainty at each point
        """
        # Calculate signal derivatives
        signal_diff = np.abs(np.diff(signal))
        signal_diff = np.append(signal_diff, signal_diff[-1])  # Maintain length

        # Normalize to [0, 1]
        if signal_diff.max() > 0:
            signal_diff_norm = signal_diff / signal_diff.max()
        else:
            signal_diff_norm = signal_diff

        low_thresh, high_thresh = threshold_range
        mid_thresh = (low_thresh + high_thresh) / 2

        # Define fuzzy membership for "high change" state
        if membership_type == 'gaussian':
            sigma = (high_thresh - low_thresh) / 4
            fuzzy_membership = FuzzySetAnalyzer.gaussian_membership(
                signal_diff_norm, mid_thresh, sigma
            )
        elif membership_type == 'triangular':
            fuzzy_membership = FuzzySetAnalyzer.triangular_membership(
                signal_diff_norm, low_thresh, mid_thresh, high_thresh
            )
        elif membership_type == 'trapezoidal':
            quarter = (high_thresh - low_thresh) / 4
            fuzzy_membership = FuzzySetAnalyzer.trapezoidal_membership(
                signal_diff_norm,
                low_thresh,
                low_thresh + quarter,
                high_thresh - quarter,
                high_thresh
            )
        else:
            raise ValueError(f"Unknown membership type: {membership_type}")

        # Crisp bifurcations (traditional)
        crisp_bifurcations = np.where(signal_diff_norm > mid_thresh)[0]

        # Fuzzy bifurcations (membership > 0.5)
        fuzzy_bifurcations = []
        for idx in range(len(fuzzy_membership)):
            if fuzzy_membership[idx] > 0.5:
                fuzzy_bifurcations.append({
                    'index': idx,
                    'membership_degree': fuzzy_membership[idx],
                    'certainty': 'high' if fuzzy_membership[idx] > 0.8 else 'medium'
                })

        # Calculate uncertainty (entropy of fuzzy membership)
        uncertainty_map = -fuzzy_membership * np.log2(fuzzy_membership + 1e-10) - \
                         (1 - fuzzy_membership) * np.log2(1 - fuzzy_membership + 1e-10)

        # State classification: LOW, MEDIUM, HIGH change
        state_memberships = {
            'low': 1 - fuzzy_membership,
            'high': fuzzy_membership,
            'medium': 1 - np.abs(fuzzy_membership - 0.5) * 2  # High at 0.5, low at 0 and 1
        }

        return {
            'crisp_bifurcations': crisp_bifurcations,
            'fuzzy_bifurcations': fuzzy_bifurcations,
            'fuzzy_membership': fuzzy_membership,
            'state_memberships': state_memberships,
            'uncertainty_map': uncertainty_map,
            'mean_uncertainty': np.mean(uncertainty_map),
            'high_uncertainty_indices': np.where(uncertainty_map > 0.5)[0]
        }

    @staticmethod
    def fuzzy_similarity(signal1, signal2, method='fuzzy_correlation'):
        """
        Calculate fuzzy similarity between two signals

        Parameters:
        -----------
        signal1, signal2 : array-like
            Signals to compare
        method : str
            'fuzzy_correlation', 'fuzzy_distance', or 'possibility'

        Returns:
        --------
        similarity : float
            Fuzzy similarity measure [0, 1]
        """
        # Normalize signals to [0, 1]
        s1_norm = (signal1 - signal1.min()) / (signal1.max() - signal1.min() + 1e-10)
        s2_norm = (signal2 - signal2.min()) / (signal2.max() - signal2.min() + 1e-10)

        if method == 'fuzzy_correlation':
            # Fuzzy correlation coefficient
            numerator = np.sum(np.minimum(s1_norm, s2_norm))
            denominator = np.sum(np.maximum(s1_norm, s2_norm))
            similarity = numerator / (denominator + 1e-10)

        elif method == 'fuzzy_distance':
            # Hamming distance-based similarity
            distance = np.mean(np.abs(s1_norm - s2_norm))
            similarity = 1 - distance

        elif method == 'possibility':
            # Possibility measure
            intersection = np.minimum(s1_norm, s2_norm)
            similarity = np.max(intersection)

        else:
            raise ValueError(f"Unknown method: {method}")

        return similarity

    @staticmethod
    def fuzzy_state_classification(signal, n_states=3):
        """
        Classify signal into fuzzy states (e.g., low/medium/high activity)

        Parameters:
        -----------
        signal : array-like
            Input signal
        n_states : int
            Number of fuzzy states (typically 3-5)

        Returns:
        --------
        result : dict
            'state_memberships': Membership for each state at each time point
            'dominant_states': Most likely state at each time
            'state_transitions': Fuzzy transitions between states
        """
        # Normalize signal
        signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-10)

        # Create overlapping fuzzy states
        state_memberships = {}
        state_centers = np.linspace(0, 1, n_states)
        sigma = 1.0 / (2 * n_states)  # Overlap between states

        for i, center in enumerate(state_centers):
            state_memberships[f'state_{i}'] = FuzzySetAnalyzer.gaussian_membership(
                signal_norm, center, sigma
            )

        # Determine dominant state at each point
        membership_matrix = np.array([state_memberships[f'state_{i}']
                                      for i in range(n_states)])
        dominant_states = np.argmax(membership_matrix, axis=0)

        # Detect fuzzy transitions (high uncertainty = between states)
        max_membership = np.max(membership_matrix, axis=0)
        transition_uncertainty = 1 - max_membership
        fuzzy_transitions = np.where(transition_uncertainty > 0.3)[0]

        return {
            'state_memberships': state_memberships,
            'dominant_states': dominant_states,
            'fuzzy_transitions': fuzzy_transitions,
            'transition_uncertainty': transition_uncertainty,
            'n_transitions': len(fuzzy_transitions)
        }

    @staticmethod
    def fuzzy_cross_modal_comparison(modality_data_dict, metric='bifurcation_similarity'):
        """
        Compare multiple modalities using fuzzy similarity measures

        Parameters:
        -----------
        modality_data_dict : dict
            Dictionary mapping modality names to their signals/features
        metric : str
            'bifurcation_similarity', 'state_similarity', or 'pattern_similarity'

        Returns:
        --------
        result : dict
            'fuzzy_similarity_matrix': Pairwise fuzzy similarities
            'crisp_similarity_matrix': Defuzzified (hard) similarities
            'most_similar_pair': Modality pair with highest similarity
            'least_similar_pair': Modality pair with lowest similarity
        """
        modalities = list(modality_data_dict.keys())
        n_modalities = len(modalities)

        fuzzy_sim_matrix = np.zeros((n_modalities, n_modalities))

        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i == j:
                    fuzzy_sim_matrix[i, j] = 1.0
                elif i < j:
                    data1 = modality_data_dict[mod1].flatten()
                    data2 = modality_data_dict[mod2].flatten()

                    # Ensure same length
                    min_len = min(len(data1), len(data2))
                    data1 = data1[:min_len]
                    data2 = data2[:min_len]

                    # Calculate fuzzy similarity
                    similarity = FuzzySetAnalyzer.fuzzy_similarity(
                        data1, data2, method='fuzzy_correlation'
                    )

                    fuzzy_sim_matrix[i, j] = similarity
                    fuzzy_sim_matrix[j, i] = similarity

        # Find most and least similar pairs
        upper_tri_indices = np.triu_indices(n_modalities, k=1)
        upper_tri_values = fuzzy_sim_matrix[upper_tri_indices]

        max_idx = np.argmax(upper_tri_values)
        min_idx = np.argmin(upper_tri_values)

        most_similar_pair = (
            modalities[upper_tri_indices[0][max_idx]],
            modalities[upper_tri_indices[1][max_idx]],
            upper_tri_values[max_idx]
        )

        least_similar_pair = (
            modalities[upper_tri_indices[0][min_idx]],
            modalities[upper_tri_indices[1][min_idx]],
            upper_tri_values[min_idx]
        )

        return {
            'modalities': modalities,
            'fuzzy_similarity_matrix': fuzzy_sim_matrix,
            'most_similar_pair': most_similar_pair,
            'least_similar_pair': least_similar_pair,
            'mean_similarity': np.mean(upper_tri_values),
            'similarity_variance': np.var(upper_tri_values)
        }


# Fuzzy-Enhanced Modality-Specific Bifurcation Detection
def detect_eeg_bifurcations_fuzzy(eeg_data, sfreq=250, use_fuzzy=True, **kwargs):
    """
    EEG bifurcation detection with fuzzy logic for uncertainty handling

    Parameters:
    -----------
    eeg_data : ndarray
        EEG data (channels x time)
    sfreq : float
        Sampling frequency
    use_fuzzy : bool
        Use fuzzy logic (True) or traditional crisp detection (False)
    **kwargs : dict
        Additional parameters for fuzzy detection

    Returns:
    --------
    result : dict
        Bifurcation results with fuzzy memberships and uncertainty maps
    """
    # Traditional temporal bifurcations
    crisp_result = NonLinearAnalyzer.detect_temporal_bifurcations(
        eeg_data, sfreq=sfreq
    )

    if not use_fuzzy:
        return crisp_result

    # Add fuzzy analysis for each channel
    fuzzy_results = []
    for ch_idx in range(eeg_data.shape[0]):
        ch_signal = eeg_data[ch_idx, :]
        fuzzy_bif = FuzzySetAnalyzer.fuzzy_bifurcation_detection(
            ch_signal,
            threshold_range=kwargs.get('threshold_range', (0.1, 0.5)),
            membership_type=kwargs.get('membership_type', 'gaussian')
        )
        fuzzy_results.append(fuzzy_bif)

    # Aggregate fuzzy bifurcations across channels
    total_fuzzy_bifurcations = sum(len(fr['fuzzy_bifurcations']) for fr in fuzzy_results)
    mean_uncertainty = np.mean([fr['mean_uncertainty'] for fr in fuzzy_results])

    # Fuzzy state classification for global signal
    global_signal = np.mean(eeg_data, axis=0)
    fuzzy_states = FuzzySetAnalyzer.fuzzy_state_classification(global_signal, n_states=3)

    return {
        **crisp_result,
        'fuzzy_bifurcations_per_channel': fuzzy_results,
        'total_fuzzy_bifurcations': total_fuzzy_bifurcations,
        'mean_uncertainty': mean_uncertainty,
        'fuzzy_states': fuzzy_states,
        'high_confidence_bifurcations': sum(
            1 for fr in fuzzy_results
            for fb in fr['fuzzy_bifurcations']
            if fb['membership_degree'] > 0.8
        )
    }


def detect_meg_bifurcations_fuzzy(meg_data, sfreq=1000, use_fuzzy=True, **kwargs):
    """
    MEG bifurcation detection with fuzzy logic

    Parameters:
    -----------
    meg_data : ndarray
        MEG data (channels x time)
    sfreq : float
        Sampling frequency
    use_fuzzy : bool
        Use fuzzy logic
    **kwargs : dict
        Additional fuzzy parameters

    Returns:
    --------
    result : dict
        MEG bifurcation results with fuzzy analysis
    """
    # Traditional MEG bifurcations
    crisp_result = NonLinearAnalyzer.detect_meg_bifurcations(meg_data, sfreq=sfreq)

    if not use_fuzzy:
        return crisp_result

    # Fuzzy analysis on global signal
    global_signal = np.mean(meg_data, axis=0)
    fuzzy_global = FuzzySetAnalyzer.fuzzy_bifurcation_detection(
        global_signal,
        threshold_range=kwargs.get('threshold_range', (0.1, 0.5)),
        membership_type=kwargs.get('membership_type', 'gaussian')
    )

    # Fuzzy C-means clustering for spatial patterns
    n_samples = meg_data.shape[1]
    window_size = min(1000, n_samples // 10)
    n_windows = n_samples // window_size

    spatial_patterns = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_mean = np.mean(meg_data[:, start:end], axis=1)
        spatial_patterns.append(window_mean)

    if len(spatial_patterns) > 3:
        spatial_patterns_array = np.array(spatial_patterns)
        fuzzy_clusters = FuzzySetAnalyzer.fuzzy_cmeans_clustering(
            spatial_patterns_array,
            n_clusters=min(3, len(spatial_patterns)),
            m=2
        )
    else:
        fuzzy_clusters = None

    return {
        **crisp_result,
        'fuzzy_global_bifurcations': fuzzy_global,
        'fuzzy_spatial_clusters': fuzzy_clusters,
        'mean_uncertainty': fuzzy_global['mean_uncertainty']
    }


def detect_mri_bifurcations_fuzzy(mri_data, use_fuzzy=True, **kwargs):
    """
    MRI bifurcation detection with fuzzy logic for spatial transitions

    Parameters:
    -----------
    mri_data : ndarray
        MRI volume (3D or 2D)
    use_fuzzy : bool
        Use fuzzy logic
    **kwargs : dict
        Additional fuzzy parameters

    Returns:
    --------
    result : dict
        MRI bifurcation results with fuzzy spatial analysis
    """
    # Traditional spatial bifurcations
    crisp_result = NonLinearAnalyzer.detect_spatial_bifurcations(
        mri_data,
        method=kwargs.get('method', 'gradient')
    )

    if not use_fuzzy:
        return crisp_result

    # Flatten for fuzzy analysis
    mri_flat = mri_data.flatten()

    # Fuzzy bifurcation detection on intensity transitions
    fuzzy_spatial = FuzzySetAnalyzer.fuzzy_bifurcation_detection(
        mri_flat,
        threshold_range=kwargs.get('threshold_range', (0.1, 0.5)),
        membership_type=kwargs.get('membership_type', 'gaussian')
    )

    # Fuzzy state classification (tissue types: CSF/Gray/White matter)
    fuzzy_tissue_states = FuzzySetAnalyzer.fuzzy_state_classification(
        mri_flat,
        n_states=kwargs.get('n_tissue_states', 3)
    )

    # Fuzzy C-means for tissue segmentation
    sample_size = min(10000, len(mri_flat))
    sample_indices = np.random.choice(len(mri_flat), sample_size, replace=False)
    mri_sample = mri_flat[sample_indices].reshape(-1, 1)

    fuzzy_segmentation = FuzzySetAnalyzer.fuzzy_cmeans_clustering(
        mri_sample,
        n_clusters=kwargs.get('n_clusters', 3),
        m=2
    )

    return {
        **crisp_result,
        'fuzzy_spatial_bifurcations': fuzzy_spatial,
        'fuzzy_tissue_states': fuzzy_tissue_states,
        'fuzzy_segmentation': fuzzy_segmentation,
        'mean_uncertainty': fuzzy_spatial['mean_uncertainty']
    }


def detect_fmri_bifurcations_fuzzy(fmri_data, tr=2.0, use_fuzzy=True, **kwargs):
    """
    fMRI bifurcation detection with fuzzy spatio-temporal analysis

    Parameters:
    -----------
    fmri_data : ndarray
        fMRI data (x, y, z, time)
    tr : float
        Repetition time
    use_fuzzy : bool
        Use fuzzy logic
    **kwargs : dict
        Additional fuzzy parameters

    Returns:
    --------
    result : dict
        fMRI bifurcation results with fuzzy analysis
    """
    # Traditional fMRI bifurcations
    crisp_result = NonLinearAnalyzer.detect_fmri_bifurcations(
        fmri_data,
        tr=tr,
        method=kwargs.get('method', 'spatio-temporal')
    )

    if not use_fuzzy:
        return crisp_result

    # Fuzzy analysis on BOLD signal
    global_bold = np.mean(fmri_data, axis=(0, 1, 2))
    fuzzy_bold = FuzzySetAnalyzer.fuzzy_bifurcation_detection(
        global_bold,
        threshold_range=kwargs.get('threshold_range', (0.1, 0.5)),
        membership_type=kwargs.get('membership_type', 'gaussian')
    )

    # Fuzzy state classification for activation levels
    fuzzy_activation_states = FuzzySetAnalyzer.fuzzy_state_classification(
        global_bold,
        n_states=kwargs.get('n_states', 3)
    )

    # Fuzzy spatial clustering for activation patterns
    n_timepoints = fmri_data.shape[3]
    if n_timepoints > 5:
        # Sample spatial patterns at different time points
        sample_times = np.linspace(0, n_timepoints - 1, min(10, n_timepoints)).astype(int)
        spatial_patterns = []

        for t in sample_times:
            spatial_slice = fmri_data[:, :, :, t].flatten()
            # Subsample for efficiency
            sample_size = min(1000, len(spatial_slice))
            sample_indices = np.random.choice(len(spatial_slice), sample_size, replace=False)
            spatial_patterns.append(spatial_slice[sample_indices])

        spatial_patterns_array = np.array(spatial_patterns)
        fuzzy_spatial_clusters = FuzzySetAnalyzer.fuzzy_cmeans_clustering(
            spatial_patterns_array,
            n_clusters=min(3, len(sample_times)),
            m=2
        )
    else:
        fuzzy_spatial_clusters = None

    return {
        **crisp_result,
        'fuzzy_bold_bifurcations': fuzzy_bold,
        'fuzzy_activation_states': fuzzy_activation_states,
        'fuzzy_spatial_clusters': fuzzy_spatial_clusters,
        'mean_uncertainty': fuzzy_bold['mean_uncertainty']
    }


def detect_cmri_bifurcations_fuzzy(cmri_data, use_fuzzy=True, **kwargs):
    """
    cMRI bifurcation detection with fuzzy logic for contrast boundaries

    Parameters:
    -----------
    cmri_data : ndarray
        Contrast-enhanced MRI (3D or 4D)
    use_fuzzy : bool
        Use fuzzy logic
    **kwargs : dict
        Additional fuzzy parameters

    Returns:
    --------
    result : dict
        cMRI bifurcation results with fuzzy contrast analysis
    """
    # Traditional cMRI bifurcations
    crisp_result = NonLinearAnalyzer.detect_cmri_bifurcations(
        cmri_data,
        method=kwargs.get('method', 'all')
    )

    if not use_fuzzy:
        return crisp_result

    # Flatten for fuzzy analysis
    if cmri_data.ndim == 4:
        # 4D: temporal analysis
        global_enhancement = np.mean(cmri_data, axis=(0, 1, 2))
        fuzzy_enhancement = FuzzySetAnalyzer.fuzzy_bifurcation_detection(
            global_enhancement,
            threshold_range=kwargs.get('threshold_range', (0.1, 0.5)),
            membership_type=kwargs.get('membership_type', 'gaussian')
        )

        # Fuzzy perfusion states
        fuzzy_perfusion_states = FuzzySetAnalyzer.fuzzy_state_classification(
            global_enhancement,
            n_states=kwargs.get('n_states', 3)
        )
    else:
        # 3D: spatial analysis
        cmri_flat = cmri_data.flatten()
        fuzzy_enhancement = FuzzySetAnalyzer.fuzzy_bifurcation_detection(
            cmri_flat,
            threshold_range=kwargs.get('threshold_range', (0.1, 0.5)),
            membership_type=kwargs.get('membership_type', 'gaussian')
        )
        fuzzy_perfusion_states = None

    # Fuzzy contrast boundary detection
    middle_slice_idx = cmri_data.shape[2] // 2 if cmri_data.ndim >= 3 else 0
    if cmri_data.ndim == 4:
        middle_slice = cmri_data[:, :, middle_slice_idx, cmri_data.shape[3] // 2]
    else:
        middle_slice = cmri_data[:, :, middle_slice_idx] if cmri_data.ndim == 3 else cmri_data

    # Fuzzy C-means for contrast regions
    slice_flat = middle_slice.flatten()
    sample_size = min(5000, len(slice_flat))
    sample_indices = np.random.choice(len(slice_flat), sample_size, replace=False)
    slice_sample = slice_flat[sample_indices].reshape(-1, 1)

    fuzzy_contrast_regions = FuzzySetAnalyzer.fuzzy_cmeans_clustering(
        slice_sample,
        n_clusters=kwargs.get('n_clusters', 3),
        m=2
    )

    return {
        **crisp_result,
        'fuzzy_enhancement_bifurcations': fuzzy_enhancement,
        'fuzzy_perfusion_states': fuzzy_perfusion_states,
        'fuzzy_contrast_regions': fuzzy_contrast_regions,
        'mean_uncertainty': fuzzy_enhancement['mean_uncertainty']
    }


def detect_bifurcations_by_modality_fuzzy(data, modality_type, use_fuzzy=True, **kwargs):
    """
    Unified fuzzy bifurcation detection for all 5 modalities

    Parameters:
    -----------
    data : ndarray
        Neuroimaging data
    modality_type : str
        'EEG', 'MEG', 'MRI', 'fMRI', or 'cMRI'
    use_fuzzy : bool
        Use fuzzy logic (True) or traditional crisp detection (False)
    **kwargs : dict
        Modality-specific and fuzzy parameters

    Returns:
    --------
    result : dict
        Bifurcation results with optional fuzzy analysis
    """
    modality_type = modality_type.upper()

    if modality_type == 'EEG':
        return detect_eeg_bifurcations_fuzzy(
            data,
            sfreq=kwargs.get('sfreq', 250),
            use_fuzzy=use_fuzzy,
            **kwargs
        )
    elif modality_type == 'MEG':
        return detect_meg_bifurcations_fuzzy(
            data,
            sfreq=kwargs.get('sfreq', 1000),
            use_fuzzy=use_fuzzy,
            **kwargs
        )
    elif modality_type == 'MRI':
        return detect_mri_bifurcations_fuzzy(
            data,
            use_fuzzy=use_fuzzy,
            **kwargs
        )
    elif modality_type == 'FMRI':
        return detect_fmri_bifurcations_fuzzy(
            data,
            tr=kwargs.get('tr', 2.0),
            use_fuzzy=use_fuzzy,
            **kwargs
        )
    elif modality_type == 'CMRI':
        return detect_cmri_bifurcations_fuzzy(
            data,
            use_fuzzy=use_fuzzy,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown modality type: {modality_type}")


def compare_bifurcations_cross_modal_fuzzy(bifurcation_results, use_fuzzy=True):
    """
    Cross-modal bifurcation comparison with fuzzy similarity measures

    Parameters:
    -----------
    bifurcation_results : dict
        Dictionary mapping modality names to their bifurcation results
    use_fuzzy : bool
        Use fuzzy similarity (True) or traditional correlation (False)

    Returns:
    --------
    result : dict
        Cross-modal comparison with fuzzy similarity matrix
    """
    # Traditional crisp comparison
    crisp_comparison = compare_bifurcations_cross_modal(bifurcation_results)

    if not use_fuzzy:
        return crisp_comparison

    # Extract signals for fuzzy comparison
    modality_signals = {}
    for modality, result in bifurcation_results.items():
        # Try to extract a representative signal
        if 'fuzzy_membership' in result:
            modality_signals[modality] = result['fuzzy_membership']
        elif 'uncertainty_map' in result:
            modality_signals[modality] = result['uncertainty_map']
        elif isinstance(result, dict) and 'bifurcation_indices' in result:
            # Create a binary signal from bifurcation indices
            signal_length = result.get('signal_length', 1000)
            binary_signal = np.zeros(signal_length)
            if len(result['bifurcation_indices']) > 0:
                binary_signal[result['bifurcation_indices']] = 1
            modality_signals[modality] = binary_signal
        else:
            # Fallback: create a signal from bifurcation count
            modality_signals[modality] = np.array([result.get('total_bifurcations', 0)])

    # Fuzzy cross-modal comparison
    if len(modality_signals) > 1:
        fuzzy_comparison = FuzzySetAnalyzer.fuzzy_cross_modal_comparison(
            modality_signals,
            metric='bifurcation_similarity'
        )
    else:
        fuzzy_comparison = None

    return {
        **crisp_comparison,
        'fuzzy_comparison': fuzzy_comparison,
        'fuzzy_similarity_matrix': fuzzy_comparison['fuzzy_similarity_matrix'] if fuzzy_comparison else None,
        'most_similar_pair_fuzzy': fuzzy_comparison['most_similar_pair'] if fuzzy_comparison else None,
        'mean_fuzzy_similarity': fuzzy_comparison['mean_similarity'] if fuzzy_comparison else None
    }


def detect_bifurcations_transform_based(signal_data, modality_type, sfreq=None, **kwargs):
    """
    Unified transform-based bifurcation detection for all modalities

    Combines time-domain AND frequency-domain analysis for robust detection

    Parameters:
    -----------
    signal_data : ndarray
        Neuroimaging data
    modality_type : str
        'EEG', 'MEG', 'MRI', 'fMRI', or 'cMRI'
    sfreq : float
        Sampling frequency (for EEG/MEG/fMRI)
    **kwargs : dict
        Additional parameters

    Returns:
    --------
    result : dict
        Combined time-domain + frequency-domain bifurcation analysis
    """
    modality_type = modality_type.upper()

    # Time-domain bifurcations (existing methods)
    time_domain = detect_bifurcations_by_modality(
        signal_data, modality_type, **kwargs
    )

    result = {'time_domain': time_domain}

    # Add frequency-domain analysis for temporal modalities
    if modality_type in ['EEG', 'MEG', 'FMRI'] and sfreq is not None:
        # Use global signal for frequency analysis
        if signal_data.ndim == 2:  # EEG/MEG: channels x time
            global_signal = np.mean(signal_data, axis=0)
        elif signal_data.ndim == 4:  # fMRI: x,y,z,time
            global_signal = np.mean(signal_data, axis=(0, 1, 2))
        else:
            global_signal = signal_data.flatten()

        # Frequency-domain bifurcation detection (3 methods)
        freq_bifurcations = {}
        for method in ['spectral_edge', 'peak_shift', 'bandwidth_change']:
            try:
                freq_result = TransformAnalyzer.frequency_domain_bifurcation_detection(
                    global_signal, sfreq, method=method
                )
                freq_bifurcations[method] = freq_result
            except:
                freq_bifurcations[method] = None

        # Power spectral density
        try:
            psd = TransformAnalyzer.power_spectral_density(global_signal, sfreq)
            result['psd'] = psd
        except:
            result['psd'] = None

        # Wavelet transform
        try:
            wavelet = TransformAnalyzer.wavelet_transform(global_signal, sfreq)
            result['wavelet'] = wavelet
        except:
            result['wavelet'] = None

        result['frequency_domain'] = freq_bifurcations

        # Combine bifurcations from both domains
        time_bif_indices = time_domain.get('bifurcation_indices', np.array([]))
        freq_bif_indices_combined = []

        for method, freq_result in freq_bifurcations.items():
            if freq_result is not None:
                freq_bif_indices_combined.extend(freq_result['bifurcation_indices'])

        freq_bif_indices_combined = np.unique(freq_bif_indices_combined)

        result['combined_bifurcations'] = {
            'time_domain_count': len(time_bif_indices),
            'frequency_domain_count': len(freq_bif_indices_combined),
            'time_domain_indices': time_bif_indices,
            'frequency_domain_indices': freq_bif_indices_combined,
            'consensus': np.intersect1d(time_bif_indices, freq_bif_indices_combined)
        }

    return result


def compute_transform_features(signal_data, sfreq, analysis_type='full'):
    """
    Compute comprehensive transform-based features

    Parameters:
    -----------
    signal_data : array-like
        Time series data
    sfreq : float
        Sampling frequency
    analysis_type : str
        'full', 'spectral_only', 'wavelet_only', or 'fast'

    Returns:
    --------
    features : dict
        All transform-based features
    """
    features = {}

    if analysis_type in ['full', 'spectral_only', 'fast']:
        # Power spectral density
        psd_result = TransformAnalyzer.power_spectral_density(signal_data, sfreq)
        features['psd'] = psd_result
        features['dominant_freq'] = psd_result['dominant_freq']
        features['total_power'] = psd_result['total_power']
        features['band_powers'] = psd_result['band_powers']

        # Spectral ratios (clinical biomarkers)
        bp = psd_result['band_powers']
        features['theta_beta_ratio'] = bp['theta'] / (bp['beta'] + 1e-10)
        features['alpha_theta_ratio'] = bp['alpha'] / (bp['theta'] + 1e-10)

    if analysis_type in ['full', 'wavelet_only']:
        # Wavelet transform
        try:
            wavelet_result = TransformAnalyzer.wavelet_transform(signal_data, sfreq)
            features['wavelet'] = wavelet_result
            features['wavelet_entropy'] = -np.sum(
                wavelet_result['power'] * np.log2(wavelet_result['power'] + 1e-10)
            )
        except:
            features['wavelet'] = None
            features['wavelet_entropy'] = 0

    if analysis_type == 'full':
        # STFT
        stft_result = TransformAnalyzer.short_time_fourier_transform(signal_data, sfreq)
        features['stft'] = stft_result

        # Hilbert-Huang Transform
        try:
            hht_result = TransformAnalyzer.hilbert_huang_transform(signal_data, sfreq)
            features['hht'] = hht_result
            features['n_imfs'] = hht_result['n_imfs']
        except:
            features['hht'] = None
            features['n_imfs'] = 0

    return features


def compute_cross_modal_coherence(modality_data_dict, sfreq_dict):
    """
    Compute coherence between all modality pairs (for temporal modalities)

    Parameters:
    -----------
    modality_data_dict : dict
        Dictionary mapping modality names to their signals
    sfreq_dict : dict
        Dictionary mapping modality names to their sampling frequencies

    Returns:
    --------
    result : dict
        Coherence matrix and band-specific coherences
    """
    modalities = list(modality_data_dict.keys())
    n_modalities = len(modalities)

    # Initialize coherence matrix
    coherence_matrix = np.zeros((n_modalities, n_modalities))
    band_coherences = {}

    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i == j:
                coherence_matrix[i, j] = 1.0
            elif i < j:
                signal1 = modality_data_dict[mod1].flatten()
                signal2 = modality_data_dict[mod2].flatten()

                # Ensure same length
                min_len = min(len(signal1), len(signal2))
                signal1 = signal1[:min_len]
                signal2 = signal2[:min_len]

                # Use minimum sampling frequency
                sfreq = min(sfreq_dict.get(mod1, 250), sfreq_dict.get(mod2, 250))

                try:
                    coh_result = TransformAnalyzer.coherence_analysis(
                        signal1, signal2, sfreq
                    )
                    coherence_matrix[i, j] = coh_result['mean_coherence']
                    coherence_matrix[j, i] = coh_result['mean_coherence']

                    # Store band coherences
                    pair_key = f"{mod1}-{mod2}"
                    band_coherences[pair_key] = coh_result['band_coherence']
                except:
                    coherence_matrix[i, j] = 0
                    coherence_matrix[j, i] = 0

    return {
        'modalities': modalities,
        'coherence_matrix': coherence_matrix,
        'band_coherences': band_coherences,
        'mean_coherence': np.mean(coherence_matrix[np.triu_indices(n_modalities, k=1)])
    }


def compute_all_nonlinear_metrics(x, y=None, sfreq=250):
    """
    Compute all non-linear metrics for signal(s)

    Parameters:
    -----------
    x : array-like
        First signal (required)
    y : array-like
        Second signal (optional, for cross-metrics)
    sfreq : float
        Sampling frequency

    Returns:
    --------
    metrics : dict
        Dictionary of all computed metrics
    """
    metrics = {}

    # Single signal metrics
    metrics['shannon_entropy_x'] = NonLinearAnalyzer.shannon_entropy(x)
    metrics['sample_entropy_x'] = NonLinearAnalyzer.sample_entropy(x)
    metrics['approximate_entropy_x'] = NonLinearAnalyzer.approximate_entropy(x)
    metrics['higuchi_fd_x'] = NonLinearAnalyzer.higuchi_fractal_dimension(x)
    metrics['dfa_alpha_x'] = NonLinearAnalyzer.detrended_fluctuation_analysis(x)
    metrics['correlation_dim_x'] = NonLinearAnalyzer.correlation_dimension(x)
    metrics['lyapunov_x'] = NonLinearAnalyzer.lyapunov_exponent_rosenstein(x)

    # Cross-signal metrics (if y is provided)
    if y is not None:
        metrics['shannon_entropy_y'] = NonLinearAnalyzer.shannon_entropy(y)
        metrics['sample_entropy_y'] = NonLinearAnalyzer.sample_entropy(y)
        metrics['mutual_information'] = NonLinearAnalyzer.mutual_information(x, y)
        metrics['normalized_mi'] = NonLinearAnalyzer.normalized_mutual_information(x, y)

        # Phase synchronization for different frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        for band_name, (fmin, fmax) in bands.items():
            try:
                psi = NonLinearAnalyzer.phase_synchronization_index(
                    x, y, freq_band=(fmin, fmax), sfreq=sfreq
                )
                metrics[f'psi_{band_name}'] = psi
            except:
                metrics[f'psi_{band_name}'] = 0

        # Transfer entropy (bidirectional)
        metrics['transfer_entropy_y_to_x'] = NonLinearAnalyzer.transfer_entropy(x, y)
        metrics['transfer_entropy_x_to_y'] = NonLinearAnalyzer.transfer_entropy(y, x)

        # Cross-recurrence
        try:
            crqa = NonLinearAnalyzer.cross_recurrence_quantification(x, y)
            metrics['recurrence_rate'] = crqa['recurrence_rate']
        except:
            metrics['recurrence_rate'] = 0

    return metrics
