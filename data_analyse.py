import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# User-configurable parameters for frame selection
start_frame = 2000       # First frame to include in visualization
end_frame = 3000      # Last frame to include (None = all frames)
frame_step = 1        # Step size for frame selection (1 = every frame)

# Enable interactive mode for matplotlib
plt.ion()  # Turn on interactive mode

# File path
file_path = r"C:\Users\qzhou\Nextcloud\Documents\Works\Github_Projects\Channel_prediction_DNN\data\channel_estimates_x400_mission_6gnext2.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Analysis of frame pattern
print("Analyzing frame sequence pattern...")
frame_numbers = []
frame_pattern = re.compile(r'SRS Frame (\d+)')

with open(file_path, 'r') as file:
    for line in file:
        match = frame_pattern.search(line)
        if match:
            frame_numbers.append(int(match.group(1)))

frame_numbers = np.array(frame_numbers)
frame_diffs = np.diff(frame_numbers)
cycle_resets = np.where(frame_diffs < -100)[0]
frame_interval = np.bincount(np.abs(frame_diffs))[1:].argmax() + 1

# Print analysis results
print(f"Total frame entries: {len(frame_numbers)}")
print(f"Number of unique frames: {len(np.unique(frame_numbers))}")
print(f"Frame range: {min(frame_numbers)} to {max(frame_numbers)}")
print(f"Number of cycle resets detected: {len(cycle_resets)}")
print(f"Most common frame interval: {frame_interval}")

if len(cycle_resets) > 0:
    print(f"Average frames per cycle: {np.mean(np.diff(np.append([0], cycle_resets))):.2f}")
    print(f"Detected cycle reset after frames: {frame_numbers[cycle_resets]}")
    print(f"Reset to frames: {frame_numbers[cycle_resets+1]}")

# Process the full dataset with continuous timeline
print("\nProcessing full dataset for visualization...")
frames = []
subcarriers = []
magnitudes = []
timeline_indices = []
current_frame = None
cycle_count = 0
prev_frame = None

with open(file_path, 'r') as file:
    for line in file:
        # Match frame header
        frame_match = re.match(r'SRS Frame (\d+)', line)
        if frame_match:
            current_frame = int(frame_match.group(1))
            
            # Detect cycle boundaries
            if prev_frame is not None and prev_frame > current_frame:
                cycle_count += 1
                print(f"Cycle transition detected: {prev_frame} → {current_frame} (Cycle {cycle_count})")
                
            prev_frame = current_frame
            
            # Calculate continuous timeline index
            timeline_index = current_frame + (cycle_count * (max(frame_numbers) + 1))
            continue
        
        # Match subcarrier data
        sc_match = re.match(r'Sc (\d+): Re = (-?\d+), Im = (-?\d+)', line)
        if sc_match and current_frame is not None:
            sc_num = int(sc_match.group(1))
            re_val = int(sc_match.group(2))
            im_val = int(sc_match.group(3))
            
            # Calculate magnitude
            magnitude = np.sqrt(re_val**2 + im_val**2)
            
            # Store data with timeline index
            frames.append(current_frame)
            subcarriers.append(sc_num)
            magnitudes.append(magnitude)
            timeline_indices.append(timeline_index)

# Convert to numpy arrays
frames = np.array(frames)
subcarriers = np.array(subcarriers)
magnitudes = np.array(magnitudes)
timeline_indices = np.array(timeline_indices)

# Normalize magnitudes to 0-1 range
min_mag = np.min(magnitudes)
max_mag = np.max(magnitudes)
norm_magnitudes = (magnitudes - min_mag) / (max_mag - min_mag)

print(f"\nVisualization statistics:")
print(f"Total data points: {len(magnitudes)}")
print(f"Continuous timeline range: {min(timeline_indices)} to {max(timeline_indices)}")
print(f"Number of frames on timeline: {len(np.unique(timeline_indices))}")
print(f"Number of subcarriers: {len(np.unique(subcarriers))}")
print(f"Magnitude range: {min_mag:.2f} to {max_mag:.2f}")

# Filter data based on frame range selection
if end_frame is None:
    end_frame = max(timeline_indices)

# Create a mask for the selected frame range
frame_mask = ((timeline_indices >= start_frame) & 
              (timeline_indices <= end_frame) & 
              ((timeline_indices - start_frame) % frame_step == 0))

# Apply the filter to all data arrays
timeline_indices_filtered = timeline_indices[frame_mask]
subcarriers_filtered = subcarriers[frame_mask]
norm_magnitudes_filtered = norm_magnitudes[frame_mask]

print(f"\nFrame range selection:")
print(f"Original frame range: {min(timeline_indices)} to {max(timeline_indices)}")
print(f"Selected frame range: {start_frame} to {end_frame}, step: {frame_step}")
print(f"Number of frames in selection: {len(np.unique(timeline_indices_filtered))}")
print(f"Number of data points in selection: {len(timeline_indices_filtered)}")

# After processing all your data, create interactive 3D visualization
print("\nCreating interactive 3D visualization...")

# Create a figure with a 3D axis, optimized for interaction
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a rectangular grid for the mesh using filtered data
unique_timelines = np.sort(np.unique(timeline_indices_filtered))
unique_subcarriers = np.sort(np.unique(subcarriers_filtered))
X, Y = np.meshgrid(unique_timelines, unique_subcarriers)
Z = np.full(X.shape, np.nan)

# Fill Z values using filtered data
timeline_map = {t: i for i, t in enumerate(unique_timelines)}
subcarrier_map = {s: i for i, s in enumerate(unique_subcarriers)}
for t, sc, mag in zip(timeline_indices_filtered, subcarriers_filtered, norm_magnitudes_filtered):
    Z[subcarrier_map[sc], timeline_map[t]] = mag

# Handle NaN values
mask = np.isnan(Z)
Z[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Z[~mask]) if np.any(~mask) else 0

# Create the interactive surface plot
title_suffix = f" (Frames {start_frame} to {end_frame}, Step {frame_step})"
surface = ax.plot_surface(X, Y, Z, 
                         cmap='viridis',
                         edgecolor='none',
                         alpha=0.8,
                         rstride=2,  # Reduce points for faster rendering
                         cstride=2,  # Reduce points for faster rendering
                         antialiased=True,
                         shade=True)

# Set labels and title
ax.set_xlabel('Timeline (Continuous Frame Index)', fontsize=12)
ax.set_ylabel('Subcarrier Index', fontsize=12)
ax.set_zlabel('Normalized Magnitude (0-1)', fontsize=12)
ax.set_title(f' Channel Estimates{title_suffix}\n', fontsize=14)

# Add colorbar
cbar = fig.colorbar(surface, ax=ax, shrink=0.6)
cbar.set_label('Normalized Magnitude')

# Set initial view
ax.view_init(elev=30, azim=45)

# Add annotation with interaction instructions
fig.text(0.02, 0.02, "Interaction:\n• Click+drag: Rotate\n• Right-click+drag: Zoom\n• Middle-click+drag: Pan", 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Make the layout tight
plt.tight_layout()

# Show the figure in interactive mode
plt.draw()
plt.pause(0.1)  # Small pause to ensure the figure shows up

# Keep the program running to allow interaction
print("Interactive plot is now active. Close the figure window to continue.")
plt.show(block=True)

# Create 2D heatmap for better pattern visualization using filtered data
plt.figure(figsize=(16, 8))
heatmap_data = np.full((len(unique_subcarriers), len(unique_timelines)), np.nan)

# Create mapping from actual indices to array positions for heatmap
timeline_map = {idx: pos for pos, idx in enumerate(unique_timelines)}
subcarrier_map = {idx: pos for pos, idx in enumerate(unique_subcarriers)}

# Fill in the heatmap with filtered data
for t, sc, m in zip(timeline_indices_filtered, subcarriers_filtered, norm_magnitudes_filtered):
    heatmap_data[subcarrier_map[sc], timeline_map[t]] = m

# Handle NaN values in heatmap
mask = np.isnan(heatmap_data)
heatmap_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), heatmap_data[~mask]) if np.any(~mask) else 0

# Plot heatmap
plt.imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Normalized Magnitude')
plt.xlabel('Timeline (Continuous Frame Index)')
plt.ylabel('Subcarrier Index')
plt.title(f'Channel Estimates Heatmap{title_suffix}')
plt.tight_layout()
plt.show()