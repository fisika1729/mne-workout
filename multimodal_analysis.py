#!/usr/bin/env python3
"""
Multi-modal MRI/EEG Data Analysis Tool using MNE-Python
Supports multiple formats and correlation analysis between MRI and EEG data
"""

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import mne
import numpy as np
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MultiModalAnalyzer:
    """Advanced application for multi-modal MRI/EEG data analysis"""

    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Modal MRI/EEG Data Analyzer")
        self.root.geometry("1400x900")

        # Data storage
        self.eeg_data = None
        self.mri_data = None
        self.eeg_file_path = None
        self.mri_file_path = None
        self.correlation_results = {}

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load EEG Data", command=self.load_eeg_data)
        file_menu.add_command(label="Load MRI Data", command=self.load_mri_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Compute Correlations", command=self.compute_correlations)
        analysis_menu.add_command(label="Source Localization", command=self.source_localization)

        # Top control panel
        control_panel = ttk.Frame(self.root, padding="10")
        control_panel.pack(side=tk.TOP, fill=tk.X)

        # EEG controls
        eeg_frame = ttk.LabelFrame(control_panel, text="EEG Data", padding="5")
        eeg_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(eeg_frame, text="Load EEG",
                  command=self.load_eeg_data).pack(side=tk.LEFT, padx=5)
        self.eeg_label = ttk.Label(eeg_frame, text="No EEG data loaded")
        self.eeg_label.pack(side=tk.LEFT, padx=10)

        # MRI controls
        mri_frame = ttk.LabelFrame(control_panel, text="MRI Data", padding="5")
        mri_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(mri_frame, text="Load MRI",
                  command=self.load_mri_data).pack(side=tk.LEFT, padx=5)
        self.mri_label = ttk.Label(mri_frame, text="No MRI data loaded")
        self.mri_label.pack(side=tk.LEFT, padx=10)

        # Analysis controls
        analysis_frame = ttk.LabelFrame(control_panel, text="Analysis", padding="5")
        analysis_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(analysis_frame, text="Correlate",
                  command=self.compute_correlations).pack(side=tk.LEFT, padx=5)

        # Notebook for different visualization tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: EEG Data
        self.eeg_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.eeg_tab, text="EEG Data")

        # Tab 2: MRI Data
        self.mri_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.mri_tab, text="MRI Data")

        # Tab 3: Correlation Analysis
        self.corr_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.corr_tab, text="Correlation Analysis")

        # Tab 4: Source Localization
        self.source_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.source_tab, text="Source Localization")

        # Tab 5: Info
        self.info_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.info_tab, text="Data Info")

        # Info text widget
        info_scroll = ttk.Scrollbar(self.info_tab)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_text = tk.Text(self.info_tab, wrap=tk.WORD,
                                yscrollcommand=info_scroll.set, padx=10, pady=10)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        info_scroll.config(command=self.info_text.yview)

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_eeg_data(self):
        """Load EEG data from various formats"""
        initial_dir = os.path.expanduser("~")

        file_path = filedialog.askopenfilename(
            title="Select EEG file",
            initialdir=initial_dir,
            filetypes=[
                ("All EEG formats", "*.set *.fif *.edf *.bdf *.vhdr *.cnt"),
                ("EEGLAB files", "*.set"),
                ("FIF files (MNE)", "*.fif"),
                ("EDF files", "*.edf"),
                ("BDF files", "*.bdf"),
                ("BrainVision files", "*.vhdr"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.eeg_file_path = Path(file_path)
        self.load_eeg_file()

    def load_eeg_file(self):
        """Load the EEG file based on format"""
        if not self.eeg_file_path or not self.eeg_file_path.exists():
            messagebox.showerror("Error", "File not found!")
            return

        try:
            self.status_bar.config(text=f"Loading EEG: {self.eeg_file_path.name}...")
            self.root.update()

            # Detect format and load accordingly
            ext = self.eeg_file_path.suffix.lower()

            if ext == '.set':
                self.eeg_data = mne.io.read_raw_eeglab(str(self.eeg_file_path),
                                                       preload=True, verbose=False)
            elif ext == '.fif':
                self.eeg_data = mne.io.read_raw_fif(str(self.eeg_file_path),
                                                    preload=True, verbose=False)
            elif ext == '.edf':
                self.eeg_data = mne.io.read_raw_edf(str(self.eeg_file_path),
                                                    preload=True, verbose=False)
            elif ext == '.bdf':
                self.eeg_data = mne.io.read_raw_bdf(str(self.eeg_file_path),
                                                    preload=True, verbose=False)
            elif ext == '.vhdr':
                self.eeg_data = mne.io.read_raw_brainvision(str(self.eeg_file_path),
                                                            preload=True, verbose=False)
            else:
                raise ValueError(f"Unsupported format: {ext}")

            # Update UI
            self.eeg_label.config(text=f"EEG: {self.eeg_file_path.name}")
            self.status_bar.config(text=f"Loaded EEG: {self.eeg_file_path.name}")

            # Display info and plot
            self.update_info_display()
            self.plot_eeg_data()

            messagebox.showinfo("Success",
                f"EEG data loaded!\n"
                f"Format: {ext}\n"
                f"Channels: {len(self.eeg_data.ch_names)}\n"
                f"Duration: {self.eeg_data.times[-1]:.2f} seconds\n"
                f"Sampling rate: {self.eeg_data.info['sfreq']} Hz"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load EEG file:\n{str(e)}")
            self.status_bar.config(text="Error loading EEG file")

    def load_mri_data(self):
        """Load MRI data from various formats"""
        initial_dir = os.path.expanduser("~")

        file_path = filedialog.askopenfilename(
            title="Select MRI file",
            initialdir=initial_dir,
            filetypes=[
                ("All MRI formats", "*.nii *.nii.gz *.mgz *.mgh"),
                ("NIfTI files", "*.nii *.nii.gz"),
                ("FreeSurfer files", "*.mgz *.mgh"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.mri_file_path = Path(file_path)
        self.load_mri_file()

    def load_mri_file(self):
        """Load the MRI file"""
        if not self.mri_file_path or not self.mri_file_path.exists():
            messagebox.showerror("Error", "File not found!")
            return

        try:
            self.status_bar.config(text=f"Loading MRI: {self.mri_file_path.name}...")
            self.root.update()

            # Load MRI using nibabel (if available) or MNE
            try:
                import nibabel as nib
                mri_img = nib.load(str(self.mri_file_path))
                self.mri_data = mri_img.get_fdata()
                mri_affine = mri_img.affine

                self.mri_info = {
                    'shape': self.mri_data.shape,
                    'affine': mri_affine,
                    'header': mri_img.header,
                    'voxel_size': mri_img.header.get_zooms()
                }
            except ImportError:
                # Fallback to MNE for certain formats
                self.mri_data = mne.read_surface(str(self.mri_file_path))
                self.mri_info = {'type': 'surface', 'shape': self.mri_data[0].shape}

            # Update UI
            self.mri_label.config(text=f"MRI: {self.mri_file_path.name}")
            self.status_bar.config(text=f"Loaded MRI: {self.mri_file_path.name}")

            # Display info and plot
            self.update_info_display()
            self.plot_mri_data()

            messagebox.showinfo("Success",
                f"MRI data loaded!\n"
                f"Shape: {self.mri_data.shape}\n"
                f"Data type: {self.mri_data.dtype}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load MRI file:\n{str(e)}")
            self.status_bar.config(text="Error loading MRI file")

    def plot_eeg_data(self):
        """Plot EEG data"""
        if self.eeg_data is None:
            return

        # Clear previous plots
        for widget in self.eeg_tab.winfo_children():
            widget.destroy()

        # Create control frame
        control_frame = ttk.Frame(self.eeg_tab)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Duration (s):").pack(side=tk.LEFT, padx=5)
        duration_var = tk.StringVar(value="10")
        ttk.Entry(control_frame, textvariable=duration_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Refresh Plot",
                  command=lambda: self.plot_eeg_data()).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Plot PSD",
                  command=self.plot_eeg_psd).pack(side=tk.LEFT, padx=5)

        # Create plot frame
        plot_frame = ttk.Frame(self.eeg_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Plot raw data
        try:
            duration = float(duration_var.get())
        except:
            duration = 10.0

        fig = self.eeg_data.plot(
            duration=duration,
            n_channels=min(20, len(self.eeg_data.ch_names)),
            scalings='auto',
            title=f'EEG Data: {self.eeg_file_path.name}',
            show=False,
            block=False
        )

    def plot_eeg_psd(self):
        """Plot EEG power spectral density"""
        if self.eeg_data is None:
            return

        # Clear EEG tab
        for widget in self.eeg_tab.winfo_children():
            widget.destroy()

        plot_frame = ttk.Frame(self.eeg_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Compute and plot PSD
        spectrum = self.eeg_data.compute_psd(fmax=50)
        fig = spectrum.plot(average=True, show=False)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()

    def plot_mri_data(self):
        """Plot MRI data slices"""
        if self.mri_data is None:
            return

        # Clear previous plots
        for widget in self.mri_tab.winfo_children():
            widget.destroy()

        plot_frame = ttk.Frame(self.mri_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with multiple slices
        fig = Figure(figsize=(12, 4))

        # Plot three orthogonal slices
        if len(self.mri_data.shape) >= 3:
            # Sagittal
            ax1 = fig.add_subplot(131)
            slice_idx = self.mri_data.shape[0] // 2
            ax1.imshow(self.mri_data[slice_idx, :, :].T, cmap='gray', origin='lower')
            ax1.set_title(f'Sagittal (slice {slice_idx})')
            ax1.axis('off')

            # Coronal
            ax2 = fig.add_subplot(132)
            slice_idx = self.mri_data.shape[1] // 2
            ax2.imshow(self.mri_data[:, slice_idx, :].T, cmap='gray', origin='lower')
            ax2.set_title(f'Coronal (slice {slice_idx})')
            ax2.axis('off')

            # Axial
            ax3 = fig.add_subplot(133)
            slice_idx = self.mri_data.shape[2] // 2
            ax3.imshow(self.mri_data[:, :, slice_idx].T, cmap='gray', origin='lower')
            ax3.set_title(f'Axial (slice {slice_idx})')
            ax3.axis('off')

        fig.suptitle(f'MRI: {self.mri_file_path.name}')
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()

    def compute_correlations(self):
        """Compute correlations between EEG and MRI data"""
        if self.eeg_data is None or self.mri_data is None:
            messagebox.showwarning("Warning",
                "Please load both EEG and MRI data first!")
            return

        try:
            self.status_bar.config(text="Computing correlations...")
            self.root.update()

            # Extract features from EEG
            eeg_features = self.extract_eeg_features()

            # Extract features from MRI
            mri_features = self.extract_mri_features()

            # Compute various correlation metrics
            self.correlation_results = {
                'eeg_features': eeg_features,
                'mri_features': mri_features,
                'timestamp': np.datetime64('now')
            }

            # Spectral-spatial correlation
            self.compute_spectral_spatial_correlation(eeg_features, mri_features)

            # Display results
            self.display_correlation_results()
            self.plot_correlation_results()

            self.status_bar.config(text="Correlation analysis complete")
            messagebox.showinfo("Success", "Correlation analysis completed!")

        except Exception as e:
            messagebox.showerror("Error", f"Correlation analysis failed:\n{str(e)}")
            self.status_bar.config(text="Error in correlation analysis")

    def extract_eeg_features(self):
        """Extract features from EEG data"""
        features = {}

        # Get data
        data = self.eeg_data.get_data()
        sfreq = self.eeg_data.info['sfreq']

        # Frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        # Compute power in each band for each channel
        for band_name, (fmin, fmax) in bands.items():
            band_power = []
            for ch_data in data:
                freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=int(sfreq*2))
                idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                band_power.append(np.mean(psd[idx_band]))
            features[f'power_{band_name}'] = np.array(band_power)

        # Overall statistics
        features['mean_amplitude'] = np.mean(np.abs(data), axis=1)
        features['std_amplitude'] = np.std(data, axis=1)
        features['variance'] = np.var(data, axis=1)

        # Connectivity (simplified - correlation between channels)
        features['connectivity_matrix'] = np.corrcoef(data)

        return features

    def extract_mri_features(self):
        """Extract features from MRI data"""
        features = {}

        # Flatten and normalize MRI data
        mri_flat = self.mri_data.flatten()
        mri_nonzero = mri_flat[mri_flat > np.percentile(mri_flat, 5)]  # Remove background

        # Statistical features
        features['mean_intensity'] = np.mean(mri_nonzero)
        features['std_intensity'] = np.std(mri_nonzero)
        features['variance'] = np.var(mri_nonzero)
        features['skewness'] = stats.skew(mri_nonzero)
        features['kurtosis'] = stats.kurtosis(mri_nonzero)

        # Regional features (divide into regions)
        if len(self.mri_data.shape) >= 3:
            # Simple regional division
            regions = self.divide_mri_regions(self.mri_data)
            features['regional_means'] = [np.mean(r[r > 0]) if np.any(r > 0) else 0
                                         for r in regions]
            features['regional_stds'] = [np.std(r[r > 0]) if np.any(r > 0) else 0
                                        for r in regions]

        # PCA components
        if len(self.mri_data.shape) >= 3:
            # Sample slices for PCA
            middle_slice = self.mri_data[:, :, self.mri_data.shape[2]//2]
            middle_slice_flat = middle_slice.flatten().reshape(-1, 1)
            if len(middle_slice_flat) > 10:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(middle_slice_flat)
                pca = PCA(n_components=min(10, len(scaled_data)))
                features['pca_components'] = pca.fit_transform(scaled_data[:1000])  # Sample

        return features

    def divide_mri_regions(self, mri_data, n_regions=8):
        """Divide MRI into regions"""
        regions = []

        if len(mri_data.shape) >= 3:
            # Divide into octants
            x_mid, y_mid, z_mid = [s // 2 for s in mri_data.shape]

            regions.append(mri_data[:x_mid, :y_mid, :z_mid])
            regions.append(mri_data[:x_mid, :y_mid, z_mid:])
            regions.append(mri_data[:x_mid, y_mid:, :z_mid])
            regions.append(mri_data[:x_mid, y_mid:, z_mid:])
            regions.append(mri_data[x_mid:, :y_mid, :z_mid])
            regions.append(mri_data[x_mid:, :y_mid, z_mid:])
            regions.append(mri_data[x_mid:, y_mid:, :z_mid])
            regions.append(mri_data[x_mid:, y_mid:, z_mid:])

        return regions

    def compute_spectral_spatial_correlation(self, eeg_features, mri_features):
        """Compute correlation between EEG spectral features and MRI spatial features"""
        correlations = {}

        # Correlate EEG power bands with MRI regional intensities
        if 'regional_means' in mri_features:
            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                power_key = f'power_{band}'
                if power_key in eeg_features:
                    eeg_power_mean = np.mean(eeg_features[power_key])
                    mri_regional_mean = np.mean(mri_features['regional_means'])

                    # Compute correlation
                    if len(mri_features['regional_means']) == len(eeg_features[power_key]):
                        corr, p_value = stats.pearsonr(
                            eeg_features[power_key][:len(mri_features['regional_means'])],
                            mri_features['regional_means']
                        )
                        correlations[f'{band}_regional_corr'] = {
                            'correlation': corr,
                            'p_value': p_value
                        }

        self.correlation_results['spectral_spatial'] = correlations

    def display_correlation_results(self):
        """Display correlation results in info tab"""
        results_text = []
        results_text.append("=" * 80)
        results_text.append("MULTI-MODAL CORRELATION ANALYSIS")
        results_text.append("=" * 80)
        results_text.append("")

        if 'eeg_features' in self.correlation_results:
            results_text.append("EEG FEATURES:")
            results_text.append("-" * 80)
            eeg_feat = self.correlation_results['eeg_features']

            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                key = f'power_{band}'
                if key in eeg_feat:
                    results_text.append(f"  {band.capitalize()} power (mean): "
                                      f"{np.mean(eeg_feat[key]):.4f}")

            if 'mean_amplitude' in eeg_feat:
                results_text.append(f"  Mean amplitude: {np.mean(eeg_feat['mean_amplitude']):.4f}")
            results_text.append("")

        if 'mri_features' in self.correlation_results:
            results_text.append("MRI FEATURES:")
            results_text.append("-" * 80)
            mri_feat = self.correlation_results['mri_features']

            for key in ['mean_intensity', 'std_intensity', 'skewness', 'kurtosis']:
                if key in mri_feat:
                    results_text.append(f"  {key.replace('_', ' ').capitalize()}: "
                                      f"{mri_feat[key]:.4f}")
            results_text.append("")

        if 'spectral_spatial' in self.correlation_results:
            results_text.append("SPECTRAL-SPATIAL CORRELATIONS:")
            results_text.append("-" * 80)
            corr_data = self.correlation_results['spectral_spatial']

            for key, values in corr_data.items():
                if isinstance(values, dict):
                    results_text.append(f"  {key}:")
                    results_text.append(f"    Correlation: {values['correlation']:.4f}")
                    results_text.append(f"    P-value: {values['p_value']:.4e}")
                    if values['p_value'] < 0.05:
                        results_text.append(f"    ** Statistically significant **")
            results_text.append("")

        results_text.append("=" * 80)
        results_text.append("INTERPRETATION:")
        results_text.append("-" * 80)
        results_text.append("Correlation values close to +1 or -1 indicate strong linear relationships")
        results_text.append("P-values < 0.05 suggest statistically significant correlations")
        results_text.append("These correlations help identify potential functional-structural relationships")
        results_text.append("=" * 80)

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "\n".join(results_text))

    def plot_correlation_results(self):
        """Plot correlation results"""
        # Clear correlation tab
        for widget in self.corr_tab.winfo_children():
            widget.destroy()

        plot_frame = ttk.Frame(self.corr_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(14, 8))

        # Plot 1: EEG band powers
        ax1 = fig.add_subplot(231)
        if 'eeg_features' in self.correlation_results:
            bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
            powers = [np.mean(self.correlation_results['eeg_features'].get(f'power_{b}', [0]))
                     for b in bands]
            ax1.bar(bands, powers, color='steelblue')
            ax1.set_title('EEG Band Powers')
            ax1.set_ylabel('Mean Power')
            ax1.tick_params(axis='x', rotation=45)

        # Plot 2: MRI regional intensities
        ax2 = fig.add_subplot(232)
        if 'mri_features' in self.correlation_results:
            if 'regional_means' in self.correlation_results['mri_features']:
                regions = range(len(self.correlation_results['mri_features']['regional_means']))
                intensities = self.correlation_results['mri_features']['regional_means']
                ax2.bar(regions, intensities, color='coral')
                ax2.set_title('MRI Regional Intensities')
                ax2.set_xlabel('Region')
                ax2.set_ylabel('Mean Intensity')

        # Plot 3: Correlation coefficients
        ax3 = fig.add_subplot(233)
        if 'spectral_spatial' in self.correlation_results:
            corr_data = self.correlation_results['spectral_spatial']
            labels = []
            values = []
            colors = []

            for key, val in corr_data.items():
                if isinstance(val, dict) and 'correlation' in val:
                    labels.append(key.replace('_regional_corr', ''))
                    values.append(val['correlation'])
                    colors.append('green' if val['p_value'] < 0.05 else 'gray')

            if labels:
                ax3.barh(labels, values, color=colors)
                ax3.set_title('Spectral-Spatial Correlations')
                ax3.set_xlabel('Correlation Coefficient')
                ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax3.set_xlim([-1, 1])

        # Plot 4: EEG connectivity matrix
        ax4 = fig.add_subplot(234)
        if 'eeg_features' in self.correlation_results:
            if 'connectivity_matrix' in self.correlation_results['eeg_features']:
                conn_matrix = self.correlation_results['eeg_features']['connectivity_matrix']
                im = ax4.imshow(conn_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax4.set_title('EEG Channel Connectivity')
                ax4.set_xlabel('Channel')
                ax4.set_ylabel('Channel')
                plt.colorbar(im, ax=ax4)

        # Plot 5: Feature comparison scatter
        ax5 = fig.add_subplot(235)
        if 'eeg_features' in self.correlation_results and 'mri_features' in self.correlation_results:
            eeg_feat = self.correlation_results['eeg_features']
            mri_feat = self.correlation_results['mri_features']

            if 'power_alpha' in eeg_feat and 'regional_means' in mri_feat:
                n_points = min(len(eeg_feat['power_alpha']), len(mri_feat['regional_means']))
                if n_points > 0:
                    ax5.scatter(eeg_feat['power_alpha'][:n_points],
                              mri_feat['regional_means'][:n_points],
                              alpha=0.6, color='purple')
                    ax5.set_title('Alpha Power vs MRI Intensity')
                    ax5.set_xlabel('Alpha Power')
                    ax5.set_ylabel('MRI Regional Intensity')

                    # Add trend line
                    if n_points > 1:
                        z = np.polyfit(eeg_feat['power_alpha'][:n_points],
                                      mri_feat['regional_means'][:n_points], 1)
                        p = np.poly1d(z)
                        ax5.plot(eeg_feat['power_alpha'][:n_points],
                               p(eeg_feat['power_alpha'][:n_points]),
                               "r--", alpha=0.8, linewidth=2)

        # Plot 6: Summary statistics
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        summary_text = "CORRELATION SUMMARY\n\n"

        if 'spectral_spatial' in self.correlation_results:
            sig_count = sum(1 for v in self.correlation_results['spectral_spatial'].values()
                          if isinstance(v, dict) and v.get('p_value', 1) < 0.05)
            total_count = len(self.correlation_results['spectral_spatial'])
            summary_text += f"Significant correlations:\n{sig_count}/{total_count}\n\n"

            # Find strongest correlation
            max_corr = max((v['correlation'] for v in self.correlation_results['spectral_spatial'].values()
                          if isinstance(v, dict)), default=0)
            summary_text += f"Strongest correlation:\n{max_corr:.4f}\n\n"

        summary_text += "Green bars indicate\nstatistically significant\ncorrelations (p < 0.05)"
        ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Multi-Modal Correlation Analysis', fontsize=14, fontweight='bold')
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()

        # Switch to correlation tab
        self.notebook.select(self.corr_tab)

    def source_localization(self):
        """Perform source localization (simplified demonstration)"""
        if self.eeg_data is None:
            messagebox.showwarning("Warning", "Please load EEG data first!")
            return

        messagebox.showinfo("Source Localization",
            "Source localization requires MRI-based forward models.\n\n"
            "For full source localization:\n"
            "1. Use MNE-Python's make_forward_solution()\n"
            "2. Compute inverse operator with make_inverse_operator()\n"
            "3. Apply inverse solution to data\n\n"
            "This requires proper MRI segmentation and BEM models.\n"
            "See MNE-Python documentation for complete workflow.")

    def update_info_display(self):
        """Update the info display with current data information"""
        info_text = []
        info_text.append("=" * 80)
        info_text.append("LOADED DATA INFORMATION")
        info_text.append("=" * 80)
        info_text.append("")

        if self.eeg_data is not None:
            info_text.append("EEG DATA:")
            info_text.append("-" * 80)
            info_text.append(f"File: {self.eeg_file_path.name}")
            info_text.append(f"Path: {self.eeg_file_path.parent}")
            info_text.append(f"Channels: {len(self.eeg_data.ch_names)}")
            info_text.append(f"Sampling rate: {self.eeg_data.info['sfreq']} Hz")
            info_text.append(f"Duration: {self.eeg_data.times[-1]:.2f} seconds")
            info_text.append(f"Samples: {len(self.eeg_data.times)}")
            info_text.append("")

        if self.mri_data is not None:
            info_text.append("MRI DATA:")
            info_text.append("-" * 80)
            info_text.append(f"File: {self.mri_file_path.name}")
            info_text.append(f"Path: {self.mri_file_path.parent}")
            info_text.append(f"Shape: {self.mri_data.shape}")
            info_text.append(f"Data type: {self.mri_data.dtype}")
            if hasattr(self, 'mri_info') and 'voxel_size' in self.mri_info:
                info_text.append(f"Voxel size: {self.mri_info['voxel_size']}")
            info_text.append("")

        info_text.append("=" * 80)

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "\n".join(info_text))


def main():
    """Main function"""
    current_dir = Path.cwd()
    print(f"Multi-Modal MRI/EEG Analyzer")
    print(f"Current directory: {current_dir}")
    print("=" * 60)

    root = tk.Tk()
    app = MultiModalAnalyzer(root)

    print("\nApplication started!")
    print("\nSupported formats:")
    print("  EEG: .set, .fif, .edf, .bdf, .vhdr")
    print("  MRI: .nii, .nii.gz, .mgz, .mgh")
    print("\nLoad both EEG and MRI data, then use 'Correlate' to find similarities.")

    root.mainloop()


if __name__ == "__main__":
    main()
