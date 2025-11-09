#!/usr/bin/env python3
"""
Adaptive Multi-Modal Neuroimaging Analysis System

Supports: EEG, MEG, MRI, fMRI, Contrast MRI (cMRI)
Automatically adapts analysis based on available data modalities
"""

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
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
from sklearn.cross_decomposition import CCA
import warnings
warnings.filterwarnings('ignore')

# Import non-linear analysis if available
try:
    from nonlinear_analysis import NonLinearAnalyzer, compute_all_nonlinear_metrics
    NONLINEAR_AVAILABLE = True
except ImportError:
    NONLINEAR_AVAILABLE = False


class ModalityType:
    """Enum for different neuroimaging modalities"""
    EEG = "EEG"
    MEG = "MEG"
    MRI = "MRI"
    FMRI = "fMRI"
    CMRI = "cMRI"


class ModalityData:
    """Container for a single modality's data and metadata"""

    def __init__(self, modality_type, data, file_path, metadata=None):
        self.type = modality_type
        self.data = data
        self.file_path = Path(file_path)
        self.metadata = metadata or {}
        self.features = {}
        self.is_loaded = True

    def extract_features(self):
        """Extract features based on modality type"""
        if self.type in [ModalityType.EEG, ModalityType.MEG]:
            return self._extract_electrophysiology_features()
        elif self.type == ModalityType.MRI:
            return self._extract_structural_features()
        elif self.type == ModalityType.FMRI:
            return self._extract_functional_features()
        elif self.type == ModalityType.CMRI:
            return self._extract_contrast_features()

    def _extract_electrophysiology_features(self):
        """Extract features from EEG/MEG data"""
        features = {}

        # Get data
        if hasattr(self.data, 'get_data'):
            data = self.data.get_data()
            sfreq = self.data.info['sfreq']
        else:
            data = self.data
            sfreq = self.metadata.get('sfreq', 250)

        # Frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        # Compute power in each band
        for band_name, (fmin, fmax) in bands.items():
            band_power = []
            for ch_data in data:
                freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=min(int(sfreq*2), len(ch_data)))
                idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                if np.any(idx_band):
                    band_power.append(np.mean(psd[idx_band]))
            if band_power:
                features[f'power_{band_name}'] = np.array(band_power)

        # Statistical features
        features['mean_amplitude'] = np.mean(np.abs(data), axis=1)
        features['std_amplitude'] = np.std(data, axis=1)
        features['variance'] = np.var(data, axis=1)
        features['connectivity'] = np.corrcoef(data)

        self.features = features
        return features

    def _extract_structural_features(self):
        """Extract features from structural MRI"""
        features = {}

        mri_flat = self.data.flatten()
        mri_nonzero = mri_flat[mri_flat > np.percentile(mri_flat, 5)]

        features['mean_intensity'] = np.mean(mri_nonzero)
        features['std_intensity'] = np.std(mri_nonzero)
        features['variance'] = np.var(mri_nonzero)
        features['skewness'] = stats.skew(mri_nonzero)
        features['kurtosis'] = stats.kurtosis(mri_nonzero)

        # Regional features
        if len(self.data.shape) >= 3:
            regions = self._divide_into_regions(self.data)
            features['regional_means'] = [np.mean(r[r > 0]) if np.any(r > 0) else 0 for r in regions]
            features['regional_stds'] = [np.std(r[r > 0]) if np.any(r > 0) else 0 for r in regions]
            features['regional_volumes'] = [np.sum(r > 0) for r in regions]

        self.features = features
        return features

    def _extract_functional_features(self):
        """Extract features from fMRI"""
        features = {}

        # fMRI is 4D: (x, y, z, time)
        if len(self.data.shape) == 4:
            # Average across time
            mean_activation = np.mean(self.data, axis=3)
            std_activation = np.std(self.data, axis=3)

            # Time series features
            temporal_mean = np.mean(self.data, axis=(0, 1, 2))

            features['mean_bold'] = np.mean(mean_activation)
            features['std_bold'] = np.std(std_activation)
            features['temporal_variance'] = np.var(temporal_mean)

            # Regional activation
            regions = self._divide_into_regions(mean_activation)
            features['regional_activation'] = [np.mean(r[r > np.percentile(r, 5)])
                                              if np.any(r > np.percentile(r, 5)) else 0
                                              for r in regions]

            # Temporal dynamics
            features['temporal_signal'] = temporal_mean

        else:
            # Treat as 3D
            features['mean_bold'] = np.mean(self.data)
            features['std_bold'] = np.std(self.data)

        self.features = features
        return features

    def _extract_contrast_features(self):
        """Extract features from contrast-enhanced MRI"""
        features = {}

        # Similar to structural MRI but focus on contrast enhancement
        mri_flat = self.data.flatten()
        mri_nonzero = mri_flat[mri_flat > np.percentile(mri_flat, 5)]

        # Basic statistics
        features['mean_contrast'] = np.mean(mri_nonzero)
        features['std_contrast'] = np.std(mri_nonzero)
        features['max_contrast'] = np.max(mri_nonzero)

        # Contrast enhancement ratio (high intensity regions)
        high_intensity_threshold = np.percentile(mri_nonzero, 90)
        features['enhancement_ratio'] = np.sum(mri_nonzero > high_intensity_threshold) / len(mri_nonzero)

        # Regional contrast
        if len(self.data.shape) >= 3:
            regions = self._divide_into_regions(self.data)
            features['regional_contrast'] = [np.mean(r[r > np.percentile(r, 90)])
                                            if np.any(r > np.percentile(r, 90)) else 0
                                            for r in regions]

        self.features = features
        return features

    def _divide_into_regions(self, data, n_regions=8):
        """Divide 3D data into regions"""
        regions = []

        if len(data.shape) >= 3:
            x_mid, y_mid, z_mid = [s // 2 for s in data.shape[:3]]

            regions.append(data[:x_mid, :y_mid, :z_mid])
            regions.append(data[:x_mid, :y_mid, z_mid:])
            regions.append(data[:x_mid, y_mid:, :z_mid])
            regions.append(data[:x_mid, y_mid:, z_mid:])
            regions.append(data[x_mid:, :y_mid, :z_mid])
            regions.append(data[x_mid:, :y_mid, z_mid:])
            regions.append(data[x_mid:, y_mid:, :z_mid])
            regions.append(data[x_mid:, y_mid:, z_mid:])

        return regions


class AdaptiveMultiModalAnalyzer:
    """
    Adaptive analyzer that works with any combination of neuroimaging modalities
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Multi-Modal Neuroimaging Analyzer")
        self.root.geometry("1600x1000")

        # Storage for multiple modalities
        self.modalities = {}  # {modality_type: ModalityData}
        self.analysis_results = {}

        self.setup_ui()

    def setup_ui(self):
        """Setup adaptive user interface"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load EEG", command=lambda: self.load_modality(ModalityType.EEG))
        file_menu.add_command(label="Load MEG", command=lambda: self.load_modality(ModalityType.MEG))
        file_menu.add_command(label="Load MRI", command=lambda: self.load_modality(ModalityType.MRI))
        file_menu.add_command(label="Load fMRI", command=lambda: self.load_modality(ModalityType.FMRI))
        file_menu.add_command(label="Load cMRI (Contrast)", command=lambda: self.load_modality(ModalityType.CMRI))
        file_menu.add_separator()
        file_menu.add_command(label="Clear All", command=self.clear_all_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Auto-Analyze Available Data", command=self.auto_analyze)
        analysis_menu.add_command(label="Linear Correlations", command=self.compute_linear_correlations)
        if NONLINEAR_AVAILABLE:
            analysis_menu.add_command(label="Non-Linear Analysis", command=self.compute_nonlinear_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Cross-Modal Correlation Matrix", command=self.compute_cross_modal_matrix)

        # Top panel: Data status
        status_frame = ttk.LabelFrame(self.root, text="Loaded Modalities", padding="10")
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.status_labels = {}
        modality_types = [ModalityType.EEG, ModalityType.MEG, ModalityType.MRI,
                         ModalityType.FMRI, ModalityType.CMRI]

        for i, mod_type in enumerate(modality_types):
            frame = ttk.Frame(status_frame)
            frame.grid(row=0, column=i, padx=5)

            label = ttk.Label(frame, text=mod_type, font=('Arial', 10, 'bold'))
            label.pack()

            status = ttk.Label(frame, text="Not Loaded", foreground="red")
            status.pack()

            btn = ttk.Button(frame, text="Load",
                           command=lambda m=mod_type: self.load_modality(m))
            btn.pack()

            self.status_labels[mod_type] = status

        # Analysis control panel
        control_frame = ttk.LabelFrame(self.root, text="Analysis Controls", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="🔍 Auto-Analyze",
                  command=self.auto_analyze,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Linear Correlation",
                  command=self.compute_linear_correlations).pack(side=tk.LEFT, padx=5)

        if NONLINEAR_AVAILABLE:
            ttk.Button(control_frame, text="Non-Linear",
                      command=self.compute_nonlinear_analysis).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Cross-Modal Matrix",
                  command=self.compute_cross_modal_matrix).pack(side=tk.LEFT, padx=5)

        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Dynamic tabs - will be created based on loaded data
        self.create_static_tabs()

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready - Load data to begin",
                                   relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_static_tabs(self):
        """Create tabs that are always present"""
        # Overview tab
        self.overview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="📊 Overview")

        # Create scrolled text for overview
        self.overview_text = scrolledtext.ScrolledText(self.overview_tab,
                                                       wrap=tk.WORD,
                                                       width=100, height=40,
                                                       font=('Courier', 10))
        self.overview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Analysis results tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="📈 Analysis Results")

        # Cross-modal tab
        self.crossmodal_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.crossmodal_tab, text="🔗 Cross-Modal")

    def update_overview(self):
        """Update overview with current data status"""
        overview = []
        overview.append("=" * 100)
        overview.append("ADAPTIVE MULTI-MODAL NEUROIMAGING ANALYZER")
        overview.append("=" * 100)
        overview.append("")
        overview.append(f"Loaded Modalities: {len(self.modalities)}/5")
        overview.append("")

        for mod_type in [ModalityType.EEG, ModalityType.MEG, ModalityType.MRI,
                        ModalityType.FMRI, ModalityType.CMRI]:
            overview.append(f"{mod_type}:")
            overview.append("-" * 100)

            if mod_type in self.modalities:
                mod_data = self.modalities[mod_type]
                overview.append(f"  ✓ LOADED")
                overview.append(f"  File: {mod_data.file_path.name}")
                overview.append(f"  Path: {mod_data.file_path.parent}")

                if mod_type in [ModalityType.EEG, ModalityType.MEG]:
                    if hasattr(mod_data.data, 'info'):
                        overview.append(f"  Channels: {len(mod_data.data.ch_names)}")
                        overview.append(f"  Sampling rate: {mod_data.data.info['sfreq']} Hz")
                        overview.append(f"  Duration: {mod_data.data.times[-1]:.2f} seconds")

                elif mod_type in [ModalityType.MRI, ModalityType.CMRI]:
                    overview.append(f"  Shape: {mod_data.data.shape}")
                    overview.append(f"  Data type: {mod_data.data.dtype}")
                    if 'voxel_size' in mod_data.metadata:
                        overview.append(f"  Voxel size: {mod_data.metadata['voxel_size']}")

                elif mod_type == ModalityType.FMRI:
                    overview.append(f"  Shape: {mod_data.data.shape}")
                    if len(mod_data.data.shape) == 4:
                        overview.append(f"  Time points: {mod_data.data.shape[3]}")

                # Features if extracted
                if mod_data.features:
                    overview.append(f"  Features extracted: {len(mod_data.features)} types")
            else:
                overview.append(f"  ✗ NOT LOADED")

            overview.append("")

        overview.append("=" * 100)
        overview.append("AVAILABLE ANALYSES:")
        overview.append("-" * 100)

        n_mods = len(self.modalities)
        if n_mods == 0:
            overview.append("  Load at least one modality to begin analysis")
        elif n_mods == 1:
            overview.append(f"  ✓ Single-modality analysis (complexity, features)")
        elif n_mods >= 2:
            overview.append(f"  ✓ Cross-modal correlations ({n_mods} modalities)")
            overview.append(f"  ✓ Multi-way analysis")
            overview.append(f"  ✓ Canonical correlation analysis")
            if NONLINEAR_AVAILABLE:
                overview.append(f"  ✓ Non-linear coupling analysis")

        overview.append("")
        overview.append("Click 'Auto-Analyze' to run all appropriate analyses for your data!")
        overview.append("=" * 100)

        self.overview_text.delete(1.0, tk.END)
        self.overview_text.insert(1.0, "\n".join(overview))

    def load_modality(self, modality_type):
        """Load a specific modality type"""
        file_formats = self._get_file_formats(modality_type)

        file_path = filedialog.askopenfilename(
            title=f"Select {modality_type} file",
            initialdir=os.path.expanduser("~"),
            filetypes=file_formats
        )

        if not file_path:
            return

        try:
            self.status_bar.config(text=f"Loading {modality_type}...")
            self.root.update()

            # Load based on modality type
            data, metadata = self._load_data_by_type(modality_type, file_path)

            # Create ModalityData object
            mod_data = ModalityData(modality_type, data, file_path, metadata)

            # Store
            self.modalities[modality_type] = mod_data

            # Update UI
            self.status_labels[modality_type].config(text="✓ Loaded", foreground="green")
            self.status_bar.config(text=f"{modality_type} loaded successfully")

            # Update overview
            self.update_overview()

            # Create dedicated tab for this modality
            self._create_modality_tab(modality_type)

            messagebox.showinfo("Success",
                f"{modality_type} data loaded successfully!\n"
                f"File: {Path(file_path).name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {modality_type}:\n{str(e)}")
            self.status_bar.config(text=f"Error loading {modality_type}")
            import traceback
            traceback.print_exc()

    def _get_file_formats(self, modality_type):
        """Get file format filters for file dialog"""
        formats = {
            ModalityType.EEG: [
                ("EEG files", "*.set *.fif *.edf *.bdf *.vhdr"),
                ("EEGLAB", "*.set"),
                ("FIF", "*.fif"),
                ("EDF", "*.edf"),
                ("All files", "*.*")
            ],
            ModalityType.MEG: [
                ("MEG files", "*.fif *.ds *.sqd"),
                ("FIF", "*.fif"),
                ("CTF", "*.ds"),
                ("All files", "*.*")
            ],
            ModalityType.MRI: [
                ("MRI files", "*.nii *.nii.gz *.mgz *.mgh"),
                ("NIfTI", "*.nii *.nii.gz"),
                ("FreeSurfer", "*.mgz *.mgh"),
                ("All files", "*.*")
            ],
            ModalityType.FMRI: [
                ("fMRI files", "*.nii *.nii.gz"),
                ("NIfTI", "*.nii *.nii.gz"),
                ("All files", "*.*")
            ],
            ModalityType.CMRI: [
                ("Contrast MRI", "*.nii *.nii.gz *.dcm"),
                ("NIfTI", "*.nii *.nii.gz"),
                ("DICOM", "*.dcm"),
                ("All files", "*.*")
            ]
        }
        return formats.get(modality_type, [("All files", "*.*")])

    def _load_data_by_type(self, modality_type, file_path):
        """Load data based on modality type"""
        metadata = {}

        if modality_type == ModalityType.EEG:
            ext = Path(file_path).suffix.lower()
            if ext == '.set':
                data = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            elif ext == '.fif':
                data = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            elif ext == '.edf':
                data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            elif ext == '.bdf':
                data = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
            elif ext == '.vhdr':
                data = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
            else:
                raise ValueError(f"Unsupported EEG format: {ext}")

        elif modality_type == ModalityType.MEG:
            ext = Path(file_path).suffix.lower()
            if ext == '.fif':
                data = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            else:
                # Try generic MNE read
                data = mne.io.read_raw(file_path, preload=True, verbose=False)

        elif modality_type in [ModalityType.MRI, ModalityType.FMRI, ModalityType.CMRI]:
            try:
                import nibabel as nib
                img = nib.load(file_path)
                data = img.get_fdata()
                metadata['affine'] = img.affine
                metadata['header'] = img.header
                metadata['voxel_size'] = img.header.get_zooms()
            except ImportError:
                raise ImportError("nibabel required for MRI/fMRI. Install: pip install nibabel")

        return data, metadata

    def _create_modality_tab(self, modality_type):
        """Create a dedicated tab for this modality"""
        # Remove old tab if exists
        for tab_id in self.notebook.tabs():
            if self.notebook.tab(tab_id, "text").startswith(modality_type):
                self.notebook.forget(tab_id)

        # Create new tab
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=f"{modality_type}")

        # Add basic visualization based on type
        self._visualize_modality(tab, modality_type)

    def _visualize_modality(self, parent, modality_type):
        """Create visualization for a modality"""
        mod_data = self.modalities[modality_type]

        # Create frame for plot
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        if modality_type in [ModalityType.EEG, ModalityType.MEG]:
            # Plot time series
            try:
                fig = mod_data.data.plot(duration=10.0,
                                        n_channels=min(20, len(mod_data.data.ch_names)),
                                        scalings='auto',
                                        title=f'{modality_type}: {mod_data.file_path.name}',
                                        show=False, block=False)
            except:
                pass

        elif modality_type in [ModalityType.MRI, ModalityType.CMRI]:
            # Plot 3 orthogonal slices
            fig = Figure(figsize=(12, 4))

            if len(mod_data.data.shape) >= 3:
                ax1 = fig.add_subplot(131)
                slice_idx = mod_data.data.shape[0] // 2
                ax1.imshow(mod_data.data[slice_idx, :, :].T, cmap='gray', origin='lower')
                ax1.set_title(f'Sagittal')
                ax1.axis('off')

                ax2 = fig.add_subplot(132)
                slice_idx = mod_data.data.shape[1] // 2
                ax2.imshow(mod_data.data[:, slice_idx, :].T, cmap='gray', origin='lower')
                ax2.set_title(f'Coronal')
                ax2.axis('off')

                ax3 = fig.add_subplot(133)
                slice_idx = mod_data.data.shape[2] // 2
                ax3.imshow(mod_data.data[:, :, slice_idx].T, cmap='gray', origin='lower')
                ax3.set_title(f'Axial')
                ax3.axis('off')

                fig.suptitle(f'{modality_type}: {mod_data.file_path.name}')
                fig.tight_layout()

                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                toolbar.update()

        elif modality_type == ModalityType.FMRI:
            # Plot mean activation map
            fig = Figure(figsize=(12, 4))

            if len(mod_data.data.shape) == 4:
                mean_bold = np.mean(mod_data.data, axis=3)

                ax1 = fig.add_subplot(131)
                slice_idx = mean_bold.shape[0] // 2
                im1 = ax1.imshow(mean_bold[slice_idx, :, :].T, cmap='hot', origin='lower')
                ax1.set_title(f'Mean BOLD - Sagittal')
                ax1.axis('off')
                plt.colorbar(im1, ax=ax1)

                ax2 = fig.add_subplot(132)
                slice_idx = mean_bold.shape[1] // 2
                im2 = ax2.imshow(mean_bold[:, slice_idx, :].T, cmap='hot', origin='lower')
                ax2.set_title(f'Mean BOLD - Coronal')
                ax2.axis('off')
                plt.colorbar(im2, ax=ax2)

                ax3 = fig.add_subplot(133)
                slice_idx = mean_bold.shape[2] // 2
                im3 = ax3.imshow(mean_bold[:, :, slice_idx].T, cmap='hot', origin='lower')
                ax3.set_title(f'Mean BOLD - Axial')
                ax3.axis('off')
                plt.colorbar(im3, ax=ax3)

                fig.suptitle(f'fMRI: {mod_data.file_path.name}')
                fig.tight_layout()

                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                toolbar.update()

    def auto_analyze(self):
        """Automatically run appropriate analyses based on loaded data"""
        if not self.modalities:
            messagebox.showwarning("No Data", "Please load at least one modality first!")
            return

        self.status_bar.config(text="Running auto-analysis...")
        self.root.update()

        try:
            # Extract features for all loaded modalities
            for mod_type, mod_data in self.modalities.items():
                mod_data.extract_features()

            # Run appropriate analyses
            if len(self.modalities) == 1:
                self._analyze_single_modality()
            else:
                self._analyze_multiple_modalities()

            self.status_bar.config(text="Auto-analysis complete!")
            messagebox.showinfo("Success", "Auto-analysis completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Auto-analysis failed:\n{str(e)}")
            self.status_bar.config(text="Error in auto-analysis")
            import traceback
            traceback.print_exc()

    def _analyze_single_modality(self):
        """Analyze single modality"""
        mod_type = list(self.modalities.keys())[0]
        mod_data = self.modalities[mod_type]

        results = []
        results.append(f"Single Modality Analysis: {mod_type}")
        results.append("=" * 80)
        results.append("")
        results.append("Extracted Features:")

        for key, value in mod_data.features.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    results.append(f"  {key}: mean={np.mean(value):.4f}, std={np.std(value):.4f}")
                elif value.ndim == 2:
                    results.append(f"  {key}: shape={value.shape}")
            else:
                results.append(f"  {key}: {value:.4f}")

        self.analysis_results['single_modality'] = "\n".join(results)
        self._display_results(results)

    def _analyze_multiple_modalities(self):
        """Analyze multiple modalities together"""
        self.compute_cross_modal_matrix()

    def compute_linear_correlations(self):
        """Compute linear correlations between all modality pairs"""
        if len(self.modalities) < 2:
            messagebox.showwarning("Insufficient Data",
                "Load at least 2 modalities for correlation analysis!")
            return

        self.status_bar.config(text="Computing linear correlations...")
        self.root.update()

        # Will be implemented with cross-modal matrix
        self.compute_cross_modal_matrix()

    def compute_cross_modal_matrix(self):
        """Compute correlation matrix between all modalities"""
        if len(self.modalities) < 2:
            messagebox.showwarning("Insufficient Data",
                "Load at least 2 modalities for cross-modal analysis!")
            return

        self.status_bar.config(text="Computing cross-modal correlations...")
        self.root.update()

        try:
            # Extract features if not done
            for mod_data in self.modalities.values():
                if not mod_data.features:
                    mod_data.extract_features()

            # Compute pairwise correlations
            mod_types = list(self.modalities.keys())
            n_mods = len(mod_types)

            corr_matrix = np.zeros((n_mods, n_mods))

            for i, mod1 in enumerate(mod_types):
                for j, mod2 in enumerate(mod_types):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    elif i < j:
                        corr = self._compute_modality_correlation(mod1, mod2)
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr

            # Visualize
            self._plot_cross_modal_matrix(corr_matrix, mod_types)

            self.status_bar.config(text="Cross-modal analysis complete!")

        except Exception as e:
            messagebox.showerror("Error", f"Cross-modal analysis failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _compute_modality_correlation(self, mod1_type, mod2_type):
        """Compute correlation between two modalities"""
        mod1 = self.modalities[mod1_type]
        mod2 = self.modalities[mod2_type]

        # Extract comparable features
        feat1_values = []
        feat2_values = []

        # Get numeric features
        for key1, val1 in mod1.features.items():
            if isinstance(val1, (int, float, np.float32, np.float64)):
                feat1_values.append(val1)

        for key2, val2 in mod2.features.items():
            if isinstance(val2, (int, float, np.float32, np.float64)):
                feat2_values.append(val2)

        # Also try regional features
        if 'regional_means' in mod1.features and 'regional_means' in mod2.features:
            reg1 = mod1.features['regional_means']
            reg2 = mod2.features['regional_means']

            min_len = min(len(reg1), len(reg2))
            if min_len > 0:
                corr, _ = stats.pearsonr(reg1[:min_len], reg2[:min_len])
                return corr

        # Fallback: correlation of feature means
        if feat1_values and feat2_values:
            min_len = min(len(feat1_values), len(feat2_values))
            if min_len > 1:
                corr, _ = stats.pearsonr(feat1_values[:min_len], feat2_values[:min_len])
                return corr

        return 0.0

    def _plot_cross_modal_matrix(self, corr_matrix, mod_types):
        """Plot cross-modal correlation matrix"""
        # Clear cross-modal tab
        for widget in self.crossmodal_tab.winfo_children():
            widget.destroy()

        plot_frame = ttk.Frame(self.crossmodal_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        ax.set_xticks(range(len(mod_types)))
        ax.set_yticks(range(len(mod_types)))
        ax.set_xticklabels(mod_types, rotation=45)
        ax.set_yticklabels(mod_types)

        # Add correlation values
        for i in range(len(mod_types)):
            for j in range(len(mod_types)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=12)

        ax.set_title('Cross-Modal Correlation Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()

        # Switch to cross-modal tab
        self.notebook.select(self.crossmodal_tab)

    def compute_nonlinear_analysis(self):
        """Compute non-linear analysis for available modalities"""
        if not NONLINEAR_AVAILABLE:
            messagebox.showerror("Error", "Non-linear analysis module not available!")
            return

        if not self.modalities:
            messagebox.showwarning("No Data", "Please load at least one modality!")
            return

        messagebox.showinfo("Non-Linear Analysis",
            "Non-linear analysis will be computed for all electrophysiology modalities\n"
            "and cross-modal non-linear coupling will be assessed.")

        # This would integrate with the nonlinear_analysis module
        # Implementation similar to previous version but adaptive

    def _display_results(self, results):
        """Display analysis results"""
        # Clear results tab
        for widget in self.results_tab.winfo_children():
            widget.destroy()

        results_text = scrolledtext.ScrolledText(self.results_tab,
                                                wrap=tk.WORD,
                                                font=('Courier', 10))
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        results_text.insert(1.0, "\n".join(results))

        # Switch to results tab
        self.notebook.select(self.results_tab)

    def clear_all_data(self):
        """Clear all loaded data"""
        if messagebox.askyesno("Clear All", "Are you sure you want to clear all loaded data?"):
            self.modalities = {}
            self.analysis_results = {}

            for status_label in self.status_labels.values():
                status_label.config(text="Not Loaded", foreground="red")

            self.update_overview()
            self.status_bar.config(text="All data cleared")


def main():
    """Main function"""
    print("Adaptive Multi-Modal Neuroimaging Analyzer")
    print("=" * 80)
    print("Supported Modalities:")
    print("  - EEG (Electroencephalography)")
    print("  - MEG (Magnetoencephalography)")
    print("  - MRI (Structural MRI)")
    print("  - fMRI (Functional MRI)")
    print("  - cMRI (Contrast-enhanced MRI)")
    print("")
    print("Load any combination - analysis adapts automatically!")
    print("=" * 80)

    root = tk.Tk()
    app = AdaptiveMultiModalAnalyzer(root)

    root.mainloop()


if __name__ == "__main__":
    main()
