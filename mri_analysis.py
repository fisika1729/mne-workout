#!/usr/bin/env python3
"""
MRI/EEG Data Analysis Tool using MNE-Python
Reads .set format files and displays data in interactive windows
"""

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import mne
import numpy as np


class MNEDataAnalyzer:
    """Main application for MNE data analysis"""

    def __init__(self, root):
        self.root = root
        self.root.title("MNE-Python Data Analyzer")
        self.root.geometry("1200x800")

        self.raw_data = None
        self.file_path = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open .set file", command=self.load_set_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Top frame for file info and controls
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top_frame, text="Load .set File",
                  command=self.load_set_file).pack(side=tk.LEFT, padx=5)

        self.file_label = ttk.Label(top_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=20)

        # Notebook for different visualization tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Raw Data Visualization
        self.raw_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.raw_tab, text="Raw Data")

        # Tab 2: PSD Plot
        self.psd_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.psd_tab, text="Power Spectral Density")

        # Tab 3: Info Tab
        self.info_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.info_tab, text="Data Info")

        # Info text widget
        self.info_text = tk.Text(self.info_tab, wrap=tk.WORD, padx=10, pady=10)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Control frame for raw data tab
        control_frame = ttk.Frame(self.raw_tab)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Duration (s):").pack(side=tk.LEFT, padx=5)
        self.duration_var = tk.StringVar(value="10")
        ttk.Entry(control_frame, textvariable=self.duration_var,
                 width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Start (s):").pack(side=tk.LEFT, padx=5)
        self.start_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.start_var,
                 width=10).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Plot Raw Data",
                  command=self.plot_raw_data).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Plot Sensors",
                  command=self.plot_sensors).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_set_file(self):
        """Load a .set file using file dialog"""
        # Get initial directory - check common locations
        initial_dir = os.path.expanduser("~")
        if os.path.exists(os.path.join(initial_dir, "Documents")):
            initial_dir = os.path.join(initial_dir, "Documents")

        file_path = filedialog.askopenfilename(
            title="Select .set file",
            initialdir=initial_dir,
            filetypes=[
                ("EEGLAB files", "*.set"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.file_path = Path(file_path)
        self.load_data()

    def load_data(self):
        """Load the data from the selected file"""
        if not self.file_path or not self.file_path.exists():
            messagebox.showerror("Error", "File not found!")
            return

        try:
            self.status_bar.config(text=f"Loading {self.file_path.name}...")
            self.root.update()

            # Load the .set file using MNE
            self.raw_data = mne.io.read_raw_eeglab(
                str(self.file_path),
                preload=True,
                verbose=False
            )

            # Update UI
            self.file_label.config(text=f"File: {self.file_path.name}")
            self.status_bar.config(text=f"Loaded {self.file_path.name} successfully")

            # Display info
            self.display_info()

            # Auto-plot raw data
            self.plot_raw_data()

            messagebox.showinfo("Success",
                f"Data loaded successfully!\n"
                f"Channels: {len(self.raw_data.ch_names)}\n"
                f"Duration: {self.raw_data.times[-1]:.2f} seconds\n"
                f"Sampling rate: {self.raw_data.info['sfreq']} Hz"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.status_bar.config(text="Error loading file")

    def display_info(self):
        """Display information about the loaded data"""
        if self.raw_data is None:
            return

        info_text = []
        info_text.append("=" * 60)
        info_text.append("DATA INFORMATION")
        info_text.append("=" * 60)
        info_text.append(f"File: {self.file_path.name}")
        info_text.append(f"File path: {self.file_path.parent}")
        info_text.append("")
        info_text.append(f"Number of channels: {len(self.raw_data.ch_names)}")
        info_text.append(f"Sampling rate: {self.raw_data.info['sfreq']} Hz")
        info_text.append(f"Duration: {self.raw_data.times[-1]:.2f} seconds")
        info_text.append(f"Number of samples: {len(self.raw_data.times)}")
        info_text.append("")
        info_text.append("Channel names:")
        for i, ch_name in enumerate(self.raw_data.ch_names, 1):
            info_text.append(f"  {i:3d}. {ch_name}")
        info_text.append("")
        info_text.append("Channel types:")
        ch_types = {}
        for ch_type in self.raw_data.get_channel_types():
            ch_types[ch_type] = ch_types.get(ch_type, 0) + 1
        for ch_type, count in ch_types.items():
            info_text.append(f"  {ch_type}: {count}")
        info_text.append("")
        info_text.append("=" * 60)

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "\n".join(info_text))

    def plot_raw_data(self):
        """Plot raw data in the matplotlib canvas"""
        if self.raw_data is None:
            messagebox.showwarning("Warning", "Please load a file first!")
            return

        try:
            duration = float(self.duration_var.get())
            start = float(self.start_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid duration or start time!")
            return

        # Clear previous plot
        for widget in self.raw_tab.winfo_children():
            if isinstance(widget, tk.Frame) and widget != self.raw_tab.winfo_children()[0]:
                widget.destroy()

        # Create matplotlib figure
        fig = self.raw_data.plot(
            duration=duration,
            start=start,
            n_channels=min(20, len(self.raw_data.ch_names)),
            scalings='auto',
            title=f'Raw Data: {self.file_path.name}',
            show=False,
            block=False
        )

        self.status_bar.config(text=f"Plotted raw data (start={start}s, duration={duration}s)")

    def plot_sensors(self):
        """Plot sensor locations"""
        if self.raw_data is None:
            messagebox.showwarning("Warning", "Please load a file first!")
            return

        try:
            # Try to plot sensor locations
            fig = self.raw_data.plot_sensors(
                show_names=True,
                title='Sensor Locations',
                show=True,
                block=False
            )
            self.status_bar.config(text="Displayed sensor locations")
        except Exception as e:
            messagebox.showwarning("Warning",
                f"Could not plot sensor locations:\n{str(e)}\n"
                "The data may not contain location information.")

    def plot_psd(self):
        """Plot power spectral density"""
        if self.raw_data is None:
            messagebox.showwarning("Warning", "Please load a file first!")
            return

        # Clear previous plot
        for widget in self.psd_tab.winfo_children():
            widget.destroy()

        # Create frame for plot
        plot_frame = ttk.Frame(self.psd_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Compute and plot PSD
        fig = self.raw_data.compute_psd(
            fmax=50
        ).plot(
            show=False,
            average=True
        )

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()

        self.status_bar.config(text="Plotted power spectral density")

        # Switch to PSD tab
        self.notebook.select(self.psd_tab)


def main():
    """Main function"""
    # Check if running from correct directory
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    print(f"Script location: {Path(__file__).parent.absolute()}")

    # Create main window
    root = tk.Tk()
    app = MNEDataAnalyzer(root)

    print("\nMNE-Python Data Analyzer started!")
    print("Please use the 'Load .set File' button to load your data.")
    print(f"Supported format: EEGLAB .set files")

    root.mainloop()


if __name__ == "__main__":
    main()
