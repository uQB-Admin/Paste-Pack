#!/usr/bin/env python
"""
Automated execution script for phagocytosis analysis
"""

import os
os.chdir(r"C:\Users\meghams\OneDrive Student\UBC(1)\brianmah@student.ubc.ca - B Cell ML Work\B Cell M0 Raw files\Phagy Count Sorted Data\Combined data")

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.stats import spearmanr
from skimage import io, measure, morphology
import glob
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("PHAGOCYTOSIS EVENT ANALYSIS")
print("="*70)
print("\nLibraries imported successfully!")

# Configuration
FRAME_START = 1
FRAME_END = 97
OVERLAP_THRESHOLD = 0.3
MIN_MACROPHAGE_AREA = 50
MAX_MOVEMENT_DISTANCE = 30

# Get all TIFF files
tiff_files = sorted(glob.glob('*.tif'))
print(f"\nFound {len(tiff_files)} TIFF files:")
for f in tiff_files:
    print(f"  - {f}")

# Load TIFF function
def load_tiff_video(filepath, frame_start=1, frame_end=97):
    print(f"\nLoading {os.path.basename(filepath)}...")
    img_stack = io.imread(filepath)

    if len(img_stack.shape) == 3:
        frames = img_stack[frame_start-1:frame_end]
    elif len(img_stack.shape) == 4:
        frames = img_stack[frame_start-1:frame_end]
    else:
        raise ValueError(f"Unexpected image shape: {img_stack.shape}")

    print(f"  Loaded shape: {frames.shape}")
    return frames

def extract_channels(frames):
    if len(frames.shape) == 3:
        print("  Warning: Single channel detected. Assuming grayscale composite.")
        red_channel = frames.copy()
        blue_channel = frames.copy()
    elif len(frames.shape) == 4:
        # Check if channels are at position 1 (time, channels, height, width)
        if frames.shape[1] <= 3:
            # Format: (time, channels, height, width)
            red_channel = frames[:, 0, :, :]  # Channel 0 = red (Raji cells)
            blue_channel = frames[:, 1, :, :] if frames.shape[1] > 1 else frames[:, 0, :, :]  # Channel 1 = blue (macrophages)
        elif frames.shape[-1] <= 3:
            # Format: (time, height, width, channels)
            red_channel = frames[..., 0]
            blue_channel = frames[..., 2] if frames.shape[-1] >= 3 else frames[..., 1]
        else:
            raise ValueError(f"Cannot determine channel layout from shape: {frames.shape}")
    else:
        raise ValueError(f"Cannot extract channels from shape: {frames.shape}")

    return red_channel, blue_channel

# Load all videos
print("\n" + "="*70)
print("LOADING VIDEOS")
print("="*70)
video_data = {}
for filepath in tiff_files:
    basename = os.path.basename(filepath).replace('.tif', '')
    frames = load_tiff_video(filepath, FRAME_START, FRAME_END)
    red_ch, blue_ch = extract_channels(frames)

    parts = basename.split('_')
    size = parts[0]
    replicate = parts[1]

    video_data[basename] = {
        'frames': frames,
        'red': red_ch,
        'blue': blue_ch,
        'size': size,
        'replicate': replicate,
        'filepath': filepath
    }

print(f"\nLoaded {len(video_data)} videos successfully!")

# Visualize channels
print("\n" + "="*70)
print("CREATING CHANNEL VISUALIZATIONS")
print("="*70)
n_videos = len(video_data)
fig, axes = plt.subplots(n_videos, 3, figsize=(15, 4*n_videos))

if n_videos == 1:
    axes = axes.reshape(1, -1)

for idx, (name, data) in enumerate(sorted(video_data.items())):
    red_frame = data['red'][0]
    blue_frame = data['blue'][0]

    # Create RGB composite from red and blue channels
    composite = np.zeros((*red_frame.shape, 3), dtype=np.uint8)
    # Normalize to 0-255 range
    red_norm = ((red_frame - red_frame.min()) / (red_frame.max() - red_frame.min() + 1e-8) * 255).astype(np.uint8)
    blue_norm = ((blue_frame - blue_frame.min()) / (blue_frame.max() - blue_frame.min() + 1e-8) * 255).astype(np.uint8)
    composite[..., 0] = red_norm  # Red channel
    composite[..., 2] = blue_norm  # Blue channel

    axes[idx, 0].imshow(red_frame, cmap='Reds')
    axes[idx, 0].set_title(f"{name}\\nRed Channel (Raji Cells)")
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(blue_frame, cmap='Blues')
    axes[idx, 1].set_title(f"{name}\\nBlue Channel (Macrophages)")
    axes[idx, 1].axis('off')

    axes[idx, 2].imshow(composite)
    axes[idx, 2].set_title(f"{name}\\nComposite")
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.savefig('channel_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Channel visualization saved!")

# Macrophage Tracker Class
class MacrophageTracker:
    def __init__(self, max_distance=30):
        self.max_distance = max_distance
        self.tracks = []
        self.next_id = 0

    def detect_macrophages(self, blue_frame, min_area=50, max_area=100000):
        threshold = np.percentile(blue_frame[blue_frame > 0], 50) if np.any(blue_frame > 0) else 0
        binary = blue_frame > threshold
        binary = morphology.remove_small_objects(binary, min_size=min_area)
        binary = ndimage.binary_fill_holes(binary)
        labeled = measure.label(binary)

        macrophages = []
        try:
            regions = measure.regionprops(labeled, intensity_image=blue_frame)
            for region in regions:
                # Skip very large regions (likely artifacts or merged cells)
                if min_area <= region.area <= max_area:
                    try:
                        macrophages.append({
                            'centroid': region.centroid,
                            'area': region.area,
                            'bbox': region.bbox,
                            'intensity': region.mean_intensity,
                            'coords': region.coords
                        })
                    except (MemoryError, np.core._exceptions._ArrayMemoryError):
                        # Skip regions that cause memory errors
                        continue
        except Exception as e:
            print(f"    Warning: Error in region detection: {str(e)}")

        return macrophages

    def update_tracks(self, macrophages, frame_idx):
        if frame_idx == 0:
            for mac in macrophages:
                self.tracks.append({
                    'id': self.next_id,
                    'positions': [mac['centroid']],
                    'frames': [frame_idx],
                    'areas': [mac['area']],
                    'coords_history': [mac['coords']],
                    'active': True
                })
                mac['track_id'] = self.next_id
                self.next_id += 1
        else:
            active_tracks = [t for t in self.tracks if t['active']]
            matched_tracks = set()
            matched_macs = set()

            for mac_idx, mac in enumerate(macrophages):
                best_dist = float('inf')
                best_track = None

                for track in active_tracks:
                    if track['id'] in matched_tracks:
                        continue

                    last_pos = track['positions'][-1]
                    dist = np.sqrt((mac['centroid'][0] - last_pos[0])**2 +
                                   (mac['centroid'][1] - last_pos[1])**2)

                    if dist < best_dist and dist < self.max_distance:
                        best_dist = dist
                        best_track = track

                if best_track is not None:
                    best_track['positions'].append(mac['centroid'])
                    best_track['frames'].append(frame_idx)
                    best_track['areas'].append(mac['area'])
                    best_track['coords_history'].append(mac['coords'])
                    mac['track_id'] = best_track['id']
                    matched_tracks.add(best_track['id'])
                    matched_macs.add(mac_idx)

            for track in active_tracks:
                if track['id'] not in matched_tracks:
                    if frame_idx - track['frames'][-1] > 5:
                        track['active'] = False

            for mac_idx, mac in enumerate(macrophages):
                if mac_idx not in matched_macs:
                    self.tracks.append({
                        'id': self.next_id,
                        'positions': [mac['centroid']],
                        'frames': [frame_idx],
                        'areas': [mac['area']],
                        'coords_history': [mac['coords']],
                        'active': True
                    })
                    mac['track_id'] = self.next_id
                    self.next_id += 1

        return macrophages

# Phagocytosis Analyzer Class
class PhagocytosisAnalyzer:
    def __init__(self, overlap_threshold=0.3):
        self.overlap_threshold = overlap_threshold
        self.events = []
        self.digestion_events = []
        self.rephagocytosis_events = []

    def calculate_overlap(self, macrophage_coords, red_frame):
        # More memory-efficient threshold calculation
        try:
            # Sample only non-zero pixels if there are too many
            nonzero_pixels = red_frame[red_frame > 0]
            if len(nonzero_pixels) > 100000:
                # Sample 100k pixels for threshold calculation
                sample_indices = np.random.choice(len(nonzero_pixels), 100000, replace=False)
                red_threshold = np.percentile(nonzero_pixels[sample_indices], 30)
            elif len(nonzero_pixels) > 0:
                red_threshold = np.percentile(nonzero_pixels, 30)
            else:
                red_threshold = 0
        except (MemoryError, np.core._exceptions._ArrayMemoryError):
            # Fallback to simple mean if percentile fails
            red_threshold = red_frame.mean() if red_frame.size > 0 else 0

        overlap_count = 0
        total_mac_pixels = len(macrophage_coords)

        for coord in macrophage_coords:
            try:
                if red_frame[coord[0], coord[1]] > red_threshold:
                    overlap_count += 1
            except (IndexError, MemoryError):
                continue

        overlap_ratio = overlap_count / total_mac_pixels if total_mac_pixels > 0 else 0
        return overlap_ratio, overlap_count

    def analyze_frame(self, macrophages, red_frame, frame_idx):
        frame_events = []

        for mac in macrophages:
            overlap_ratio, overlap_pixels = self.calculate_overlap(mac['coords'], red_frame)
            mac['overlap_ratio'] = overlap_ratio
            mac['overlap_pixels'] = overlap_pixels
            mac['is_phagocytosing'] = overlap_ratio > self.overlap_threshold

            if mac['is_phagocytosing']:
                frame_events.append({
                    'frame': frame_idx,
                    'track_id': mac.get('track_id', -1),
                    'centroid': mac['centroid'],
                    'overlap_ratio': overlap_ratio,
                    'overlap_pixels': overlap_pixels
                })

        return frame_events

    def detect_events_across_time(self, all_frame_data):
        track_states = {}

        for frame_idx, frame_events in enumerate(all_frame_data):
            for event in frame_events:
                track_id = event['track_id']
                if track_id not in track_states:
                    track_states[track_id] = []
                track_states[track_id].append({
                    'frame': frame_idx,
                    'overlap_ratio': event['overlap_ratio']
                })

        for track_id, states in track_states.items():
            if len(states) < 2:
                continue

            was_phagocytosing = False
            phago_start = None

            for i, state in enumerate(states):
                is_phagocytosing = state['overlap_ratio'] > self.overlap_threshold

                if is_phagocytosing and not was_phagocytosing:
                    phago_start = state['frame']
                    previous_phago = [e for e in self.events if e['track_id'] == track_id]

                    if len(previous_phago) > 0:
                        self.rephagocytosis_events.append({
                            'track_id': track_id,
                            'frame': state['frame'],
                            'previous_event_frame': previous_phago[-1]['start_frame']
                        })

                    self.events.append({
                        'track_id': track_id,
                        'start_frame': phago_start,
                        'end_frame': None,
                        'type': 're-phagocytosis' if len(previous_phago) > 0 else 'initial'
                    })

                elif not is_phagocytosing and was_phagocytosing:
                    if phago_start is not None:
                        for event in reversed(self.events):
                            if event['track_id'] == track_id and event['end_frame'] is None:
                                event['end_frame'] = state['frame']
                                break

                        self.digestion_events.append({
                            'track_id': track_id,
                            'start_frame': phago_start,
                            'end_frame': state['frame'],
                            'duration': state['frame'] - phago_start
                        })
                        phago_start = None

                was_phagocytosing = is_phagocytosing

print("\n" + "="*70)
print("PROCESSING VIDEOS")
print("="*70)

# Process all videos
results = {}
for video_name, data in video_data.items():
    print(f"\n{video_name}:")

    tracker = MacrophageTracker(max_distance=MAX_MOVEMENT_DISTANCE)
    analyzer = PhagocytosisAnalyzer(overlap_threshold=OVERLAP_THRESHOLD)

    red_frames = data['red']
    blue_frames = data['blue']
    n_frames = len(red_frames)

    all_frame_data = []
    phago_count_per_frame = []
    overlap_ratios_per_frame = []

    for frame_idx in range(n_frames):
        if frame_idx % 20 == 0:
            print(f"  Processing frame {frame_idx+1}/{n_frames}...", end='\r')

        macrophages = tracker.detect_macrophages(blue_frames[frame_idx], min_area=MIN_MACROPHAGE_AREA)
        macrophages = tracker.update_tracks(macrophages, frame_idx)
        frame_events = analyzer.analyze_frame(macrophages, red_frames[frame_idx], frame_idx)
        all_frame_data.append(frame_events)

        unique_tracks = set([e['track_id'] for e in frame_events])
        phago_count_per_frame.append(len(unique_tracks))

        if frame_events:
            avg_overlap = np.mean([e['overlap_ratio'] for e in frame_events])
            overlap_ratios_per_frame.append(avg_overlap)
        else:
            overlap_ratios_per_frame.append(0)

    analyzer.detect_events_across_time(all_frame_data)

    results[video_name] = {
        'size': data['size'],
        'replicate': data['replicate'],
        'tracker': tracker,
        'analyzer': analyzer,
        'phago_count_per_frame': phago_count_per_frame,
        'overlap_ratios_per_frame': overlap_ratios_per_frame,
        'total_phago_events': len(analyzer.events),
        'digestion_events': len(analyzer.digestion_events),
        'rephagocytosis_events': len(analyzer.rephagocytosis_events),
        'unique_macrophages': len(tracker.tracks)
    }

    print(f"  [OK] Complete: {results[video_name]['unique_macrophages']} macrophages, " +
          f"{results[video_name]['total_phago_events']} phagocytosis events")

print("\n\n" + "="*70)
print("ALL VIDEOS PROCESSED!")
print("="*70)

# Save summary
print("\n" + "="*70)
print("CREATING SUMMARY STATISTICS")
print("="*70)
summary_data = []
for video_name, result in results.items():
    summary_data.append({
        'Video': video_name,
        'Size': result['size'],
        'Replicate': result['replicate'],
        'Unique Macrophages': result['unique_macrophages'],
        'Total Phagocytosis Events': result['total_phago_events'],
        'Digestion Events': result['digestion_events'],
        'Re-phagocytosis Events': result['rephagocytosis_events'],
        'Avg Phago/Frame': np.mean(result['phago_count_per_frame']),
        'Max Phago/Frame': np.max(result['phago_count_per_frame'])
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(['Size', 'Replicate'])
summary_df.to_csv('phagocytosis_summary.csv', index=False)
print("[OK] Summary saved to 'phagocytosis_summary.csv'")
print("\n" + summary_df.to_string(index=False))

# Create all visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

sizes = sorted(set([r['size'] for r in results.values()]))
frames = np.arange(1, FRAME_END - FRAME_START + 2)
time_minutes = (frames - 1) * 15
time_hours = time_minutes / 60
colors = {'30um': '#E74C3C', '60um': '#3498DB', '90um': '#2ECC71'}

# Plot 1: Individual replicates (frame-based)
print("Creating plot 1/9: Individual replicates (frame-based)...")
fig, axes = plt.subplots(1, len(sizes), figsize=(6*len(sizes), 5))
if len(sizes) == 1:
    axes = [axes]

for idx, size in enumerate(sizes):
    size_results = {name: res for name, res in results.items() if res['size'] == size}
    for name, res in sorted(size_results.items()):
        axes[idx].plot(frames, res['phago_count_per_frame'],
                      label=f"{res['replicate']}", alpha=0.7, linewidth=2)
    axes[idx].set_xlabel('Frame', fontsize=12)
    axes[idx].set_ylabel('Number of Phagocytosis Events', fontsize=12)
    axes[idx].set_title(f'{size} Island Size - Individual Replicates', fontsize=14, fontweight='bold')
    axes[idx].legend(title='Replicate')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phagocytosis_events_individual_replicates.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: phagocytosis_events_individual_replicates.png")

# Plot 2: IQR (frame-based)
print("Creating plot 2/9: IQR summary (frame-based)...")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for size in sizes:
    size_results = {name: res for name, res in results.items() if res['size'] == size}
    data_matrix = np.array([res['phago_count_per_frame'] for res in size_results.values()])
    median = np.median(data_matrix, axis=0)
    q25 = np.percentile(data_matrix, 25, axis=0)
    q75 = np.percentile(data_matrix, 75, axis=0)
    color = colors.get(size, '#95A5A6')
    ax.plot(frames, median, label=f'{size}', color=color, linewidth=2.5)
    ax.fill_between(frames, q25, q75, alpha=0.3, color=color)
ax.set_xlabel('Frame', fontsize=14)
ax.set_ylabel('Number of Phagocytosis Events', fontsize=14)
ax.set_title('Phagocytosis Events Over Time (Median with IQR)', fontsize=16, fontweight='bold')
ax.legend(title='Island Size', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phagocytosis_events_IQR.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: phagocytosis_events_IQR.png")

# Plot 3: Individual replicates (time-based)
print("Creating plot 3/9: Individual replicates (time-based)...")
fig, axes = plt.subplots(1, len(sizes), figsize=(6*len(sizes), 5))
if len(sizes) == 1:
    axes = [axes]

for idx, size in enumerate(sizes):
    size_results = {name: res for name, res in results.items() if res['size'] == size}
    for name, res in sorted(size_results.items()):
        axes[idx].plot(time_hours, res['phago_count_per_frame'],
                      label=f"{res['replicate']}", alpha=0.7, linewidth=2)
    axes[idx].set_xlabel('Time (hours)', fontsize=12)
    axes[idx].set_ylabel('Number of Phagocytosis Events', fontsize=12)
    axes[idx].set_title(f'{size} Island Size - Individual Replicates\\n(24-hour time course)',
                       fontsize=14, fontweight='bold')
    axes[idx].legend(title='Replicate')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(0, 24)

plt.tight_layout()
plt.savefig('phagocytosis_events_time_individual_replicates.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: phagocytosis_events_time_individual_replicates.png")

# Plot 4: IQR (time-based)
print("Creating plot 4/9: IQR summary (time-based)...")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for size in sizes:
    size_results = {name: res for name, res in results.items() if res['size'] == size}
    data_matrix = np.array([res['phago_count_per_frame'] for res in size_results.values()])
    median = np.median(data_matrix, axis=0)
    q25 = np.percentile(data_matrix, 25, axis=0)
    q75 = np.percentile(data_matrix, 75, axis=0)
    color = colors.get(size, '#95A5A6')
    ax.plot(time_hours, median, label=f'{size}', color=color, linewidth=2.5)
    ax.fill_between(time_hours, q25, q75, alpha=0.3, color=color)
ax.set_xlabel('Time (hours)', fontsize=14)
ax.set_ylabel('Number of Phagocytosis Events', fontsize=14)
ax.set_title('Phagocytosis Events Over 24 Hours (Median with IQR)', fontsize=16, fontweight='bold')
ax.legend(title='Island Size', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 24)
ax.set_xticks(np.arange(0, 25, 4))
plt.tight_layout()
plt.savefig('phagocytosis_events_time_IQR.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: phagocytosis_events_time_IQR.png")

# Plot 5: Comparison (frame vs time)
print("Creating plot 5/9: Frame vs time comparison...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for size in sizes:
    size_results = {name: res for name, res in results.items() if res['size'] == size}
    data_matrix = np.array([res['phago_count_per_frame'] for res in size_results.values()])
    median = np.median(data_matrix, axis=0)
    q25 = np.percentile(data_matrix, 25, axis=0)
    q75 = np.percentile(data_matrix, 75, axis=0)
    color = colors.get(size, '#95A5A6')
    axes[0].plot(frames, median, label=f'{size}', color=color, linewidth=2.5)
    axes[0].fill_between(frames, q25, q75, alpha=0.3, color=color)
    axes[1].plot(time_hours, median, label=f'{size}', color=color, linewidth=2.5)
    axes[1].fill_between(time_hours, q25, q75, alpha=0.3, color=color)

axes[0].set_xlabel('Frame Number', fontsize=13)
axes[0].set_ylabel('Number of Phagocytosis Events', fontsize=13)
axes[0].set_title('A) Frame-Based Analysis (Frames 1-97)', fontsize=15, fontweight='bold', loc='left')
axes[0].legend(title='Island Size', fontsize=11, loc='best')
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Time (hours)', fontsize=13)
axes[1].set_ylabel('Number of Phagocytosis Events', fontsize=13)
axes[1].set_title('B) Time-Based Analysis (0-24 hours, 15 min intervals)', fontsize=15, fontweight='bold', loc='left')
axes[1].legend(title='Island Size', fontsize=11, loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 24)
axes[1].set_xticks(np.arange(0, 25, 4))

plt.tight_layout()
plt.savefig('phagocytosis_events_frame_vs_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: phagocytosis_events_frame_vs_time_comparison.png")

# Spearman correlation
print("Creating plot 6/9: Spearman correlation (individual)...")
correlation_results = {}
for video_name, result in results.items():
    counts = result['phago_count_per_frame']
    correlation, p_value = spearmanr(frames, counts)
    correlation_results[video_name] = {
        'size': result['size'],
        'replicate': result['replicate'],
        'correlation': correlation,
        'p_value': p_value
    }

corr_df = pd.DataFrame([
    {
        'Video': name,
        'Size': data['size'],
        'Replicate': data['replicate'],
        'Spearman Correlation': data['correlation'],
        'P-value': data['p_value'],
        'Significant': 'Yes' if data['p_value'] < 0.05 else 'No'
    }
    for name, data in correlation_results.items()
]).sort_values(['Size', 'Replicate'])

corr_df.to_csv('spearman_correlation.csv', index=False)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for size in sizes:
    size_corrs = corr_df[corr_df['Size'] == size]
    x_positions = [sizes.index(size)] * len(size_corrs)
    x_jitter = np.random.normal(0, 0.05, len(size_corrs))
    ax.scatter(np.array(x_positions) + x_jitter, size_corrs['Spearman Correlation'],
               s=100, alpha=0.7, label=size, color=colors.get(size, '#95A5A6'))

ax.set_xticks(range(len(sizes)))
ax.set_xticklabels(sizes)
ax.set_xlabel('Island Size', fontsize=14)
ax.set_ylabel('Spearman Correlation Coefficient', fontsize=14)
ax.set_title('Spearman Correlation: Frame Number vs Phagocytosis Events\\n(Individual Replicates)',
             fontsize=16, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('spearman_correlation_individual.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: spearman_correlation_individual.png")

# Plot 7: Spearman correlation IQR
print("Creating plot 7/9: Spearman correlation (IQR)...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
medians = []
q25s = []
q75s = []
for size in sizes:
    size_corrs = corr_df[corr_df['Size'] == size]['Spearman Correlation'].values
    medians.append(np.median(size_corrs))
    q25s.append(np.percentile(size_corrs, 25))
    q75s.append(np.percentile(size_corrs, 75))

x_pos = np.arange(len(sizes))
ax.bar(x_pos, medians, yerr=[np.array(medians) - np.array(q25s),
                               np.array(q75s) - np.array(medians)],
       capsize=10, alpha=0.7, color=[colors.get(s, '#95A5A6') for s in sizes],
       edgecolor='black', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(sizes)
ax.set_xlabel('Island Size', fontsize=14)
ax.set_ylabel('Spearman Correlation Coefficient', fontsize=14)
ax.set_title('Spearman Correlation: Frame Number vs Phagocytosis Events\\n(Median with IQR)',
             fontsize=16, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('spearman_correlation_IQR.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: spearman_correlation_IQR.png")

# Plot 8: Digestion and re-phagocytosis
print("Creating plot 8/9: Digestion and re-phagocytosis analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

digestion_data = []
for size in sizes:
    size_results = [res for res in results.values() if res['size'] == size]
    for res in size_results:
        digestion_data.append({'Size': size, 'Count': res['digestion_events']})

digestion_df = pd.DataFrame(digestion_data)
sns.boxplot(data=digestion_df, x='Size', y='Count', ax=axes[0],
            palette=[colors.get(s, '#95A5A6') for s in sizes])
sns.swarmplot(data=digestion_df, x='Size', y='Count', ax=axes[0],
              color='black', alpha=0.5, size=8)
axes[0].set_title('Digestion Events by Island Size', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Digestion Events', fontsize=12)
axes[0].set_xlabel('Island Size', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

rephago_data = []
for size in sizes:
    size_results = [res for res in results.values() if res['size'] == size]
    for res in size_results:
        rephago_data.append({'Size': size, 'Count': res['rephagocytosis_events']})

rephago_df = pd.DataFrame(rephago_data)
sns.boxplot(data=rephago_df, x='Size', y='Count', ax=axes[1],
            palette=[colors.get(s, '#95A5A6') for s in sizes])
sns.swarmplot(data=rephago_df, x='Size', y='Count', ax=axes[1],
              color='black', alpha=0.5, size=8)
axes[1].set_title('Re-phagocytosis Events by Island Size', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Re-phagocytosis Events', fontsize=12)
axes[1].set_xlabel('Island Size', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('digestion_rephagocytosis_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: digestion_rephagocytosis_analysis.png")

# Export detailed results
print("Creating plot 9/9: Exporting detailed results...")
detailed_results = []
for video_name, result in results.items():
    for frame_idx in range(len(result['phago_count_per_frame'])):
        frame_num = frame_idx + 1
        time_min = frame_idx * 15
        time_hr = time_min / 60
        detailed_results.append({
            'Video': video_name,
            'Size': result['size'],
            'Replicate': result['replicate'],
            'Frame': frame_num,
            'Time_Minutes': time_min,
            'Time_Hours': round(time_hr, 4),
            'Phagocytosis_Events': result['phago_count_per_frame'][frame_idx],
            'Avg_Overlap_Ratio': result['overlap_ratios_per_frame'][frame_idx]
        })

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv('detailed_frame_by_frame_results.csv', index=False)
print("[OK] Saved: detailed_frame_by_frame_results.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated Files:")
print("  1. channel_visualization.png")
print("  2. phagocytosis_events_individual_replicates.png")
print("  3. phagocytosis_events_IQR.png")
print("  4. phagocytosis_events_time_individual_replicates.png")
print("  5. phagocytosis_events_time_IQR.png")
print("  6. phagocytosis_events_frame_vs_time_comparison.png")
print("  7. spearman_correlation_individual.png")
print("  8. spearman_correlation_IQR.png")
print("  9. digestion_rephagocytosis_analysis.png")
print(" 10. phagocytosis_summary.csv")
print(" 11. spearman_correlation.csv")
print(" 12. detailed_frame_by_frame_results.csv")
print("\n" + "="*70)
