import sys
import os
import numpy as np
import pydicom
from ctvbx import load_ctvbx

def verify():
    ctvbx_path = r"C:\Users\kmchord9\source\repos\ct-viewer\STL\sample06.ctvbx"
    dicom_dir = r"C:\Users\kmchord9\source\repos\ct-viewer\STL\sample06_dicom"

    print("--- CTVBX Verification ---")
    print(f"Loading CTVBX: {ctvbx_path}")
    data_ctv, info = load_ctvbx(ctvbx_path)
    
    print(f"CTV Dimensions: {info.width}x{info.height}x{info.depth}")
    print(f"CTV Voxel Size: {info.voxel_size}")
    print(f"CTV Is Signed: {info.is_signed}")
    print(f"CTV First 5 Z: {info.z_positions[:5]}")
    print(f"CTV Last 5 Z: {info.z_positions[-5:]}")
    print(f"CTV Data Range: {data_ctv.min()} ~ {data_ctv.max()}")

    print("\nLoading Original DICOM...")
    dcm_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    if not dcm_files:
        dcm_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    
    datasets = [pydicom.dcmread(f) for f in dcm_files]
    
    # Try Descending (Standard for our loader)
    datasets.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    d_z_desc = [float(ds.ImagePositionPatient[2]) for ds in datasets]
    
    # Try Ascending
    datasets_asc = sorted(datasets, key=lambda x: float(x.ImagePositionPatient[2]), reverse=False)
    d_z_asc = [float(ds.ImagePositionPatient[2]) for ds in datasets_asc]

    print(f"DICOM Dimensions: {len(datasets)} slices")
    print(f"DICOM Descending Z: {d_z_desc[:3]} ... {d_z_desc[-3:]}")
    print(f"DICOM Ascending Z: {d_z_asc[:3]} ... {d_z_asc[-3:]}")

    # Integrity Check with both
    for mode, current_datasets in [("DESCENDING", datasets), ("ASCENDING", datasets_asc)]:
        print(f"\n--- Checking Integrity ({mode} mode) ---")
        max_pixel_diff = 0
        z_match_count = 0
        
        for i in range(min(info.depth, len(current_datasets))):
            slice_ctv = data_ctv[i, :, :]
            slice_dcm = current_datasets[i].pixel_array
            
            diff = np.abs(slice_ctv.astype(np.int64) - slice_dcm.astype(np.int64))
            max_pixel_diff = max(max_pixel_diff, diff.max())
            
            z_ctv = info.z_positions[i]
            z_dcm = float(current_datasets[i].ImagePositionPatient[2])
            if abs(z_ctv - z_dcm) < 1e-4:
                z_match_count += 1
        
        print(f"Max Pixel Difference: {max_pixel_diff}")
        print(f"Z Position Matches: {z_match_count}/{info.depth}")
        if max_pixel_diff == 0 and z_match_count == info.depth:
            print(f"SUCCESS: {mode} match is PERFECT.")
        elif max_pixel_diff == 0:
            print(f"PARTIAL: Pixels match, but Z coordinates differ.")

if __name__ == "__main__":
    verify()
