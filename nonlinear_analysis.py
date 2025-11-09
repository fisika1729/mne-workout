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
