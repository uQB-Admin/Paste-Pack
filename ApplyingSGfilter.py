import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

print("=" * 70)
print("CREATING MCP_PULP PLOT (Savitzky-Golay smoothed)")
print("=" * 70)

# Read the detailed results
print("\nLoading data...")
df = pd.read_csv('detailed_frame_by_frame_results.csv')
print(f"[OK] Loaded {len(df)} rows")

# Define sizes and colors (same as box plot with brighter parrot green)
sizes = ['90um', '60um', '30um']
colors = {'90um': '#9CA9D2', '60um': '#F68F92', '30um': '#66FF66'}

# Savitzky-Golay filter parameters
SG_WINDOW = 11   # window length (must be odd); 11 frames x 15 min = 2.75 hr span
SG_POLY = 3      # polynomial order

# Create the plot with transparent background
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

for size in sizes:
    size_df = df[df['Size'] == size]

    frames = sorted(size_df['Frame'].unique())
    time_hours = sorted(size_df['Time_Hours'].unique())

    replicates = sorted(size_df['Replicate'].unique())
    data_matrix = []
    for rep in replicates:
        rep_data = size_df[size_df['Replicate'] == rep].sort_values('Frame')['Phagocytosis_Events'].values
        data_matrix.append(rep_data)

    data_matrix = np.array(data_matrix)

    median = np.median(data_matrix, axis=0)
    q25 = np.percentile(data_matrix, 25, axis=0)
    q75 = np.percentile(data_matrix, 75, axis=0)

    # Apply Savitzky-Golay filter
    window = min(SG_WINDOW, len(median) if len(median) % 2 == 1 else len(median) - 1)
    if window > SG_POLY:
        median_smooth = savgol_filter(median, window, SG_POLY)
        q25_smooth = savgol_filter(q25, window, SG_POLY)
        q75_smooth = savgol_filter(q75, window, SG_POLY)
    else:
        median_smooth, q25_smooth, q75_smooth = median, q25, q75

    color = colors.get(size, '#95A5A6')
    ax.plot(time_hours, median_smooth, color=color, linewidth=2.5)
    ax.fill_between(time_hours, q25_smooth, q75_smooth, alpha=0.3, color=color)

    print(f"{size}: {len(replicates)} replicates, {len(frames)} frames, SG window={window}, poly={SG_POLY}")

# Axis formatting (matches the formatted IQR plot style)
ax.set_xlim(0, 24)
ax.set_xticks(np.arange(0, 28, 4))

ax.set_ylim(100, 350)
ax.set_yticks(np.arange(100, 400, 50))

ax.tick_params(axis='both', which='major', labelsize=32, width=2.5, length=8)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3.0)

plt.tight_layout()
plt.savefig('MCP_PULP.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

print("\n[OK] Saved: MCP_PULP.png")
print(f"    - Savitzky-Golay filter applied (window={SG_WINDOW}, polyorder={SG_POLY})")
print("    - Transparent background")
print("    - Axis formatting matches MCP_MSP_phagocytosis_IQR_formatted")
print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
