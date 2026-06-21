import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_comparison_by_metric(filepath: str):
    """
    Generates a single figure with a 3x3 grid of plots.
    Each COLUMN represents a different metric (mean, median, std).
    Each ROW represents a different analysis (absolute value, relative mean, relative std).
    Each line within a plot represents a different run/file.
    """
    try:
        f = h5py.File(filepath, 'r')
    except FileNotFoundError:
        print(f"Error: Analysis file not found at '{filepath}'")
        return

    all_runs = list(f.keys())
    metrics_to_plot = ['mean', 'median', 'std']
    
    print("\n--- Generating Figure: Comparison by Metric ---")

    # --- Create the 3x3 Figure ---
    fig, axes = plt.subplots(3, 3, figsize=(22, 18), sharex=True)
    fig.suptitle('Detailed Convergence Analysis: Columns Separated by Metric', fontsize=20)

    colors = list(mcolors.TABLEAU_COLORS.keys())

    # --- Loop through each METRIC to define the columns ---
    for col_idx, metric_name in enumerate(metrics_to_plot):
        
        # Set the title for the top of each column
        axes[0, col_idx].set_title(f'Metric: {metric_name.upper()}', fontsize=14, weight='bold')
        
        # --- Loop through each RUN to plot the lines in the column ---
        for run_idx, run_name in enumerate(all_runs):
            run_color = colors[run_idx % len(colors)]
            
            try:
                metric_group = f[f'{run_name}/analysis_results/convergence/{metric_name}']
            except KeyError:
                print(f"Warning: Metric '{metric_name}' not found for run '{run_name}'. Skipping.")
                continue

            splits = metric_group['splits'][:]
            means = np.array(metric_group['means'][:])
            stds = np.array(metric_group['stds'][:])
            
            # --- Plot on the 3 rows for the current column ---
            
            # Row 1: Mean of the metric
            axes[0, col_idx].plot(splits, means, '.-', label=run_name, color=run_color)
            
            # Row 2: Relative change in the mean
            final_mean = means[-1]
            if abs(final_mean) > 1e-12:
                relative_change_mean = np.abs((means - final_mean) / final_mean)
                axes[1, col_idx].plot(splits, relative_change_mean, '.-', label=run_name, color=run_color)

            # Row 3: Relative change in the std dev
            final_std = stds[-1]
            if abs(final_std) > 1e-12:
                relative_change_std = np.abs((stds - final_std) / final_std)
                axes[2, col_idx].plot(splits, relative_change_std, '.-', label=run_name, color=run_color)

    # --- Configure all axes after plotting ---
    for row in range(3):
        for col in range(3):
            axes[row, col].grid(True, which='both')
            axes[row, col].legend()
            if row == 2: # Set x-label only for the bottom row
                axes[row, col].set_xlabel('Number of Samples')

    # Configure Y-labels (only for the first column)
    axes[0, 0].set_ylabel('Mean Value of Metric')
    axes[1, 0].set_ylabel('Relative Change (Fraction)')
    axes[2, 0].set_ylabel('Relative Change (Fraction)')

    # Configure scales and convergence lines
    for col in range(3):
        axes[1, col].set_yscale('log')
        axes[2, col].set_yscale('log')
        axes[1, col].axhline(0.001, color='black', linestyle=':', lw=1)
        axes[2, col].axhline(0.001, color='black', linestyle=':', lw=1)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def create_full_summary_plots(filepath: str):
    """
    Generates the original set of summary plots:
    - Fig 1: All metrics on one convergence plot.
    - Fig 2: Noise distribution comparison.
    - Fig 3: FFT spectrum comparison.
    """
    # This function is identical to the one from the previous step.
    # It is kept here to generate the other figures as requested.
    print("\n--- Generating Figure Set: Overall Summary ---")
    
    # ... (The full code for this function from the previous answer goes here) ...
    # ... (For brevity, I will omit pasting the ~100 lines of identical code) ...
    # ... Just copy the previous `create_full_summary_plots` function here ...
    # The version from the previous step is complete and correct.
    # The placeholder below shows the structure.

    try:
        f = h5py.File(filepath, 'r')
    except FileNotFoundError: return
    all_runs = list(f.keys())
    if not all_runs: f.close(); return
    
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig1.suptitle('Comparison of Convergence for Different Metrics', fontsize=16)
    fig2, ax_noise = plt.subplots(figsize=(10, 7))
    ax_noise.set_title('Comparison of Baseline Noise Distributions')
    fig3, ax_fft = plt.subplots(figsize=(10, 7))
    ax_fft.set_title('Comparison of Average FFT Spectrums')
    colors = list(mcolors.TABLEAU_COLORS.keys())
    linestyles = ['-', '--', ':']
    metrics_to_plot = ['mean', 'median', 'std']
    for i, run_name in enumerate(all_runs):
        run_group = f[run_name]
        analysis_group = run_group['analysis_results']
        run_color = colors[i % len(colors)]
        convergence_group = analysis_group['convergence']
        for j, metric_name in enumerate(metrics_to_plot):
            if metric_name not in convergence_group: continue
            metric_group = convergence_group[metric_name]
            splits = metric_group['splits'][:]
            means = np.array(metric_group['means'][:])
            stds = np.array(metric_group['stds'][:])
            metric_linestyle = linestyles[j % len(linestyles)]
            label = f'{run_name} ({metric_name})'
            ax1.plot(splits, means, label=label, color=run_color, linestyle=metric_linestyle)
            final_mean = means[-1]
            if abs(final_mean) > 1e-12:
                relative_change_mean = np.abs((means - final_mean) / final_mean)
                ax2.plot(splits, relative_change_mean, label=label, color=run_color, linestyle=metric_linestyle)
            final_std = stds[-1]
            if abs(final_std) > 1e-12:
                relative_change_std = np.abs((stds - final_std) / final_std)
                ax3.plot(splits, relative_change_std, label=label, color=run_color, linestyle=metric_linestyle)
        hist_group = analysis_group['noise_distribution']
        counts = hist_group['counts'][:]
        bin_edges = hist_group['bin_edges'][:]
        ax_noise.step(bin_edges[:-1], counts, where='post', label=run_name, alpha=0.8, linewidth=2, color=run_color)
        fft_group = analysis_group['average_fft']
        freq = fft_group['frequency_hz'][:]
        mag = fft_group['magnitude'][:]
        ax_fft.plot(freq, mag, label=run_name, color=run_color, alpha=0.8)
    ax1.set_title('Mean of Metric vs. Sample Size'); ax1.set_ylabel('Mean Value of Metric'); ax1.grid(True); ax1.legend()
    ax2.set_title('Relative Change in Mean of Metric'); ax2.set_ylabel('Relative Change (Fraction)'); ax2.set_yscale('log'); ax2.grid(True, which="both"); ax2.axhline(0.001, color='black', linestyle=':', lw=1, label='0.1% Change'); ax2.legend()
    ax3.set_title('Relative Change in Std Dev of Metric'); ax3.set_xlabel('Number of Samples in Calculation'); ax3.set_ylabel('Relative Change (Fraction)'); ax3.set_yscale('log'); ax3.grid(True, which="both"); ax3.axhline(0.001, color='black', linestyle=':', lw=1, label='0.1% Change'); ax3.legend()
    fig1.tight_layout(rect=[0, 0.03, 1, 0.96])
    ax_noise.set_xlabel('Amplitude (after baseline correction)'); ax_noise.set_ylabel('Counts per Bin'); ax_noise.grid(True); ax_noise.legend(); fig2.tight_layout()
    ax_fft.set_xlabel('Frequency (Hz)'); ax_fft.set_ylabel('Magnitude'); ax_fft.set_xscale('log'); ax_fft.set_yscale('log'); ax_fft.grid(True, which="both"); ax_fft.legend(); fig3.tight_layout()
    f.close()


if __name__ == "__main__":
    ANALYSIS_FILE_PATH = 'analysis/pmt_data_22.08.25.h5'
    
    # Make sure you have run the processing script with [np.mean, np.median, np.std]
    # to generate the necessary data.
    
    # Generate the new, detailed 3x3 figure
    create_comparison_by_metric(ANALYSIS_FILE_PATH)
    
    # Generate the original set of summary figures
    create_full_summary_plots(ANALYSIS_FILE_PATH)

    # Show all figures at once
    plt.show()