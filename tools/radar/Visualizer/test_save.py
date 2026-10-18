#!/usr/bin/env python3
"""
Test script for LRPD save functionality
"""

import os
import time
import numpy as np

def test_save_functionality():
    """Test the save functionality without GUI"""

    # Create data folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created data folder")

    # Test filename generation
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"data/LRPD_data_{timestamp}.txt"
    print(f"Would create file: {filename}")

    # Test file creation and writing
    try:
        with open(filename, 'w') as f:
            f.write("# Long Range People Detection Data Test\n")
            f.write("# Timestamp: " + timestamp + "\n")
            f.write("# FrameNum, NumPoints, NumTracks, PointCloud(x,y,z,doppler,snr), Tracks(x,y,z,vx,vy,vz)\n\n")

            # Simulate some test data
            for frame in range(1, 6):
                f.write(f"Frame: {frame}, Points: 3, Tracks: 1\n")
                f.write("Points:\n")
                for i in range(3):
                    x, y, z, doppler, snr = np.random.rand(5) * 10
                    f.write(f"  {x:.3f},{y:.3f},{z:.3f},{doppler:.3f},{snr:.3f}\n")

                f.write("Tracks:\n")
                x, y, z, vx, vy, vz = np.random.rand(6) * 5
                f.write(f"  {x:.3f},{y:.3f},{z:.3f},{vx:.3f},{vy:.3f},{vz:.3f}\n")
                f.write("\n")

        print(f"Successfully created and wrote test data to {filename}")

        # Verify file exists and has content
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
                print(f"File size: {len(content)} characters")
                print("First few lines:")
                print("\n".join(content.split("\n")[:10]))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_save_functionality()
