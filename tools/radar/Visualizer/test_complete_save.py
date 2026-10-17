#!/usr/bin/env python3
"""
Test script for complete LRPD save functionality with all parameters
"""

import os
import time
import numpy as np

def test_complete_save_functionality():
    """Test the complete save functionality with all parameters"""

    # Create data folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created data folder")

    # Test filename generation
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"data/LRPD_complete_test_{timestamp}.txt"
    print(f"Would create file: {filename}")

    # Test file creation and writing with complete data
    try:
        with open(filename, 'w') as f:
            f.write("# Long Range People Detection Data\n")
            f.write("# Start Timestamp: " + timestamp + "\n")
            f.write("# Each frame contains: Timestamp, FrameNum, NumPoints, NumTracks\n")
            f.write("# Points: X,Y,Z,Doppler,SNR,Noise,TrackIndex\n")
            f.write("# Tracks: TID,X,Y,Z,Vx,Vy,Vz,Ax,Ay,Az,G,ConfidenceLevel\n\n")

            # Simulate complete test data for 3 frames
            for frame in range(1, 4):
                # Generate timestamp with milliseconds
                frame_timestamp = time.time() + frame * 0.1
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(frame_timestamp))
                milliseconds = int((frame_timestamp % 1) * 1000)
                full_timestamp = f"{timestamp_str}.{milliseconds:03d}"

                num_points = np.random.randint(5, 15)
                num_tracks = np.random.randint(1, 4)

                f.write(f"Timestamp: {full_timestamp}, Frame: {frame}, Points: {num_points}, Tracks: {num_tracks}\n")

                # Generate complete point cloud data (7 parameters)
                f.write("Points (X,Y,Z,Doppler,SNR,Noise,TrackIndex):\n")
                for i in range(num_points):
                    x, y, z = np.random.rand(3) * 20 - 10  # -10 to 10 meters
                    doppler = np.random.rand() * 4 - 2  # -2 to 2 m/s
                    snr = np.random.rand() * 30 + 10  # 10 to 40 dB
                    noise = np.random.rand() * 10  # 0 to 10 dB
                    track_index = np.random.randint(0, num_tracks) if np.random.rand() > 0.3 else 255
                    f.write(f"  {x:.3f},{y:.3f},{z:.3f},{doppler:.3f},{snr:.3f},{noise:.3f},{track_index}\n")

                # Generate complete track data (12 parameters)
                if num_tracks > 0:
                    f.write("Tracks (TID,X,Y,Z,Vx,Vy,Vz,Ax,Ay,Az,G,ConfidenceLevel):\n")
                    for i in range(num_tracks):
                        tid = i + 1
                        x, y, z = np.random.rand(3) * 16 - 8  # -8 to 8 meters
                        vx, vy, vz = np.random.rand(3) * 6 - 3  # -3 to 3 m/s
                        ax, ay, az = np.random.rand(3) * 2 - 1  # -1 to 1 m/s²
                        g = np.random.rand() * 10  # 0 to 10
                        confidence = np.random.rand() * 0.9 + 0.1  # 0.1 to 1.0
                        f.write(f"  {tid},{x:.3f},{y:.3f},{z:.3f},{vx:.3f},{vy:.3f},{vz:.3f},{ax:.3f},{ay:.3f},{az:.3f},{g:.3f},{confidence:.3f}\n")

                f.write("\n")

        print(f"Successfully created and wrote complete test data to {filename}")

        # Verify file exists and show sample content
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
                lines = content.split("\n")
                print(f"File size: {len(content)} characters")
                print(f"Total lines: {len(lines)}")
                print("\nFirst 20 lines:")
                for i, line in enumerate(lines[:20]):
                    print("2d")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_complete_save_functionality()
