import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def merge_and_plot():
    data_dir = './data'
    
    # 1. Find all CSV files in the data directory
    # We look for numeric filenames like 1.csv, 2.csv
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not files:
        print("No CSV files found in ./data/")
        return

    # 2. Sort files numerically (1.csv, 2.csv, ... 10.csv)
    # Standard string sort would put 10.csv before 2.csv, so we use a lambda key
    try:
        files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        print("Warning: Filenames are not purely numeric. Sorting alphabetically.")
        files.sort()

    print(f"Found {len(files)} files. Merging...")

    # 3. Read and Concatenate
    dfs = []
    global_time_offset = 0
    
    for f in files:
        df = pd.read_csv(f)
        
        # Adjust time to be continuous
        # We assume the time in CSV starts near 0 for each cycle
        # We add the max time of previous file to current file
        if not dfs:
            # First file
            pass
        else:
            # Get the last timestamp of the previous dataframe
            last_t = dfs[-1]['Time'].iloc[-1]
            # Get the sampling interval (approx)
            dt = dfs[-1]['Time'].iloc[1] - dfs[-1]['Time'].iloc[0]
            # Offset current file's time
            df['Time'] = df['Time'] + last_t + dt
            
        dfs.append(df)

    full_data = pd.concat(dfs, ignore_index=True)
    
    # 4. Plot
    plt.figure(figsize=(12, 6))
    
    # Identify channel columns (exclude 'Time')
    channels = [c for c in full_data.columns if c != 'Time']
    
    for i, chan in enumerate(channels):
        # Add offset for visibility (assuming data is in uV)
        # 50 uV offset per channel usually works well
        offset = i * 100 
        plt.plot(full_data['Time'], full_data[chan] + offset, label=chan)

    plt.title("Merged EEG Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV) + Offset")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    merge_and_plot()