import h5py
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def load_results_to_dataframe(h5_file_path: str, group_name: str) -> pd.DataFrame:
    """
    Loads analysis results from a specific group in an HDF5 file into a pandas DataFrame.

    Args:
        h5_file_path (str): Path to the HDF5 results file.
        group_name (str): The name of the group (representing an original data file) to load.

    Returns:
        pd.DataFrame: A DataFrame containing the comparison metrics for all methods.
    """
    records = []
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if group_name not in f:
                print(f"Error: Group '{group_name}' not found in the HDF5 file.")
                return None
            
            analysis_group = f[group_name]
            
            for method_name in analysis_group.keys():
                method_group = analysis_group[method_name]
                
                # Create a dictionary for the current method's results
                record = {
                    'Method': method_name.replace('_', ' ') # Make names more readable
                }
                # Load all attributes (metrics) into the dictionary
                for key, value in method_group.attrs.items():
                    record[key] = value
                
                records.append(record)

    except IOError:
        print(f"Error: Could not read the file at '{h5_file_path}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    if not records:
        print(f"Warning: No methods found in group '{group_name}'.")
        return pd.DataFrame()

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(records)
    return df

def plot_comparison_charts(df: pd.DataFrame, file_identifier: str):
    """
    Generates and displays a set of bar charts to visualize the comparison results.

    Args:
        df (pd.DataFrame): The DataFrame containing the metrics.
        file_identifier (str): A string identifying the data file being analyzed.
    """
    if df.empty:
        print("Cannot generate plots from empty DataFrame.")
        return

    # Sort by F1-Score for a more logical plot layout
    df = df.sort_values('f1_score', ascending=False).reset_index()

    # --- Create the plots ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Peak Finding Method Comparison for: {file_identifier}', fontsize=16, y=0.95)

    # --- Plot 1: Accuracy Metrics (Precision, Recall, F1-Score) ---
    ax1 = axes[0]
    df[['Method', 'precision', 'recall', 'f1_score']].plot(
        x='Method', kind='bar', ax=ax1, width=0.8,
        title='Accuracy Metrics (Higher is Better)'
    )
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(title='Metric')
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)

    # --- Plot 2: Performance (Average Time) ---
    ax2 = axes[1]
    df.plot(x='Method', y='avg_time_us_per_waveform', kind='bar', ax=ax2,
            title='Performance (Lower is Better)',
            color='c', legend=False)
    ax2.set_ylabel('Average Time per Waveform (µs)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    # Set to log scale if times vary widely, otherwise linear is fine
    if df['avg_time_us_per_waveform'].max() / df['avg_time_us_per_waveform'].min() > 50:
        ax2.set_yscale('log')
        ax2.set_title('Performance (Lower is Better) - Log Scale')
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    # --- Plot 3: Physics Quality (Charge CV) ---
    ax3 = axes[2]
    df.plot(x='Method', y='charge_cv', kind='bar', ax=ax3,
            title='Physics Quality: Charge Resolution (Lower is Better)',
            color='m', legend=False)
    ax3.set_ylabel('Charge Coefficient of Variation (CV)')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    # Add a buffer to the y-axis limit
    ax3.set_ylim(0, df['charge_cv'].max() * 1.2)
    for p in ax3.patches:
        ax3.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    # --- Final Formatting ---
    plt.xlabel('Peak Finding Method')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

def display_results(h5_file_path: str):
    """
    Main function to load, display, and plot the analysis results.
    """
    # --- Discover available analysis groups in the file ---
    try:
        with h5py.File(h5_file_path, 'r') as f:
            groups = list(f.keys())
    except (IOError, FileNotFoundError):
        print(f"Error: The file '{h5_file_path}' was not found or could not be opened.")
        sys.exit(1)

    if not groups:
        print("Error: No analysis groups found in the specified HDF5 file.")
        sys.exit(1)

    # --- Let the user choose which group to analyze ---
    if len(groups) > 1:
        print("Multiple analysis results found. Please choose which one to display:")
        for i, group_name in enumerate(groups):
            print(f"  {i+1}: {group_name}")
        
        while True:
            try:
                choice = int(input(f"Enter your choice (1-{len(groups)}): ")) - 1
                if 0 <= choice < len(groups):
                    selected_group = groups[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        selected_group = groups[0]
    
    print(f"\nLoading results for '{selected_group}'...")

    # --- Load data into DataFrame ---
    results_df = load_results_to_dataframe(h5_file_path, selected_group)

    if results_df is None or results_df.empty:
        print("Could not proceed with displaying results.")
        return

    # --- Display Summary Table in Console ---
    # Define columns to show and their desired order
    display_columns = [
        'Method', 'f1_score', 'precision', 'recall', 
        'avg_time_us_per_waveform', 'charge_cv',
        'total_true_positives', 'total_false_positives', 'total_false_negatives'
    ]
    # Rename for clarity
    column_rename_map = {
        'f1_score': 'F1-Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'avg_time_us_per_waveform': 'Avg Time (µs)',
        'charge_cv': 'Charge CV'
    }
    
    # Filter and rename columns
    summary_df = results_df[display_columns].rename(columns=column_rename_map)
    summary_df = summary_df.sort_values('F1-Score', ascending=False)

    print("\n" + "="*80)
    print(f"                 PEAK FINDING METHOD COMPARISON SUMMARY for {selected_group}")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")

    # --- Generate and show plots ---
    plot_comparison_charts(results_df, selected_group)


if __name__ == "__main__":
    # --- Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Display and visualize results from the peak finding comparison pipeline."
    )
    parser.add_argument(
        'input_file',
        type=str,
        help="Path to the HDF5 file containing the analysis results (e.g., 'analysis/peak_finding_bakeoff_results.h5')."
    )
    args = parser.parse_args()
    
    # --- Run the main display function ---
    display_results(args.input_file)