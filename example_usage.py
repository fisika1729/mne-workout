#!/usr/bin/env python3
"""
Example script showing how to use MNE-Python to load and analyze .set files
This is a command-line version without GUI
"""

import os
from pathlib import Path
import mne
import matplotlib.pyplot as plt


def analyze_set_file(file_path):
    """
    Load and analyze a .set file

    Parameters:
    -----------
    file_path : str or Path
        Path to the .set file
    """
    # Convert to Path object for better path handling
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading file: {file_path}")
    print(f"File location: {file_path.parent}")
    print(f"File size: {file_path.stat().st_size / 1024:.2f} KB")
    print("-" * 60)

    # Load the .set file
    try:
        raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
        print("✓ File loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None

    # Display basic information
    print("\nDATA INFORMATION:")
    print("-" * 60)
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    print(f"Number of samples: {len(raw.times)}")

    print("\nChannel names:")
    for i, ch_name in enumerate(raw.ch_names[:10], 1):  # Show first 10
        print(f"  {i:2d}. {ch_name}")
    if len(raw.ch_names) > 10:
        print(f"  ... and {len(raw.ch_names) - 10} more channels")

    print("\nChannel types:")
    ch_types = {}
    for ch_type in raw.get_channel_types():
        ch_types[ch_type] = ch_types.get(ch_type, 0) + 1
    for ch_type, count in ch_types.items():
        print(f"  {ch_type}: {count}")

    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Plot 1: Raw data
    print("\n1. Plotting raw data...")
    fig1 = raw.plot(duration=10.0, n_channels=min(20, len(raw.ch_names)),
                    scalings='auto', title='Raw Data', show=False)
    plt.savefig('raw_data_plot.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: raw_data_plot.png")

    # Plot 2: Power Spectral Density
    print("2. Computing and plotting Power Spectral Density...")
    try:
        spectrum = raw.compute_psd(fmax=50)
        fig2 = spectrum.plot(average=True, show=False)
        plt.savefig('psd_plot.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: psd_plot.png")
    except Exception as e:
        print(f"   ✗ Could not create PSD plot: {e}")

    # Plot 3: Sensor locations (if available)
    print("3. Plotting sensor locations...")
    try:
        fig3 = raw.plot_sensors(show_names=True, show=False)
        plt.savefig('sensor_locations.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: sensor_locations.png")
    except Exception as e:
        print(f"   ✗ Could not plot sensors: {e}")
        print("   (This is normal if the data doesn't contain location info)")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

    return raw


def main():
    """Main function for command-line usage"""
    import sys

    print("MNE-Python .set File Analyzer (Command-line)")
    print("=" * 60)

    # Check if file path is provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Using file from command line: {file_path}\n")
        analyze_set_file(file_path)
    else:
        print("Usage:")
        print("  python example_usage.py <path_to_set_file>")
        print("\nExample:")
        print("  python example_usage.py /path/to/your/data.set")
        print("\nAlternatively:")
        print("  You can edit this script and set the file_path variable directly")
        print("\n" + "=" * 60)

        # Example: Uncomment and modify the line below to analyze a specific file
        # file_path = "/path/to/your/data.set"
        # analyze_set_file(file_path)

        # Or use the GUI version
        print("\nTo use the GUI version, run:")
        print("  python mri_analysis.py")


if __name__ == "__main__":
    main()
