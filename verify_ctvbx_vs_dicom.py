import sys
import os
import pydicom
import glob
from tabulate import tabulate

# Add current directory to path to ensure we use the local ctvbx library
sys.path.insert(0, os.getcwd())
import ctvbx.reader as ctvbx

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def get_first_dcm(directory):
    files = glob.glob(os.path.join(directory, "*.dcm"))
    if not files:
        return None
    return sorted(files)[0]

def count_dcm_slices(directory):
    return len(glob.glob(os.path.join(directory, "*.dcm")))

def verify_ctvbx(name, ctvbx_path, dicom_dir):
    print(f"\n=== Verifying {name} (py-ctvbx vs External DICOM) ===")
    
    # 1. Load CTVBX using our library
    try:
        vol = ctvbx.load_volume(ctvbx_path)
    except Exception as e:
        print(f"Error loading CTVBX: {e}")
        return

    # 2. Load External DICOM
    dcm_path = get_first_dcm(dicom_dir)
    if not dcm_path:
        print(f"Error: No DICOM files found in {dicom_dir}")
        return
    ds = pydicom.dcmread(dcm_path)
    slice_count = count_dcm_slices(dicom_dir)

    # 3. Prepare Comparison Data
    table = [
        ["Width / Rows", vol.width, ds.Rows, "MATCH" if vol.width == ds.Rows else "NO"],
        ["Height / Columns", vol.height, ds.Columns, "MATCH" if vol.height == ds.Columns else "NO"],
        ["Depth / Slices", vol.depth, slice_count, "MATCH" if vol.depth == slice_count else "NO"],
        ["Voxel Size / PixelSpacing", vol.voxel_size, ds.PixelSpacing[0], 
         "MATCH" if abs(vol.voxel_size - float(ds.PixelSpacing[0])) < 1e-5 else "NO"],
        ["IsSigned", vol.is_signed, ds.PixelRepresentation == 1, 
         "MATCH" if vol.is_signed == (ds.PixelRepresentation == 1) else "ALMOST (App converts to 0)"],
        ["RescaleIntercept", vol.get_parameter('RescaleIntercept', 0), getattr(ds, 'RescaleIntercept', 0), 
         "MATCH" if float(vol.get_parameter('RescaleIntercept', 0)) == float(getattr(ds, 'RescaleIntercept', 0)) else "NO"],
        ["WindowCenter (Wl)", vol.window_center, getattr(ds, 'WindowCenter', 'N/A'), "YES" if hasattr(ds, 'WindowCenter') else "CTVBX ONLY"],
        ["WindowWidth (Ww)", vol.window_width, getattr(ds, 'WindowWidth', 'N/A'), "YES" if hasattr(ds, 'WindowWidth') else "CTVBX ONLY"],
    ]

    headers = ["Attribute", "py-ctvbx (Version 2.0)", "External DICOM", "Status"]
    print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    base_dir = r"C:\Users\kmchord9\source\repos\ct-viewer\STL\checkdicom"
    
    tasks = [
        ("Test01 (Toshiba)", 
         os.path.join(base_dir, "Test01.ctvbx"), 
         os.path.join(base_dir, "Test01_dicom")),
        ("Motor (Shimadzu)", 
         os.path.join(base_dir, "motor.ctvbx"), 
         os.path.join(base_dir, "motor_dicom"))
    ]
    
    for name, ctvbx_p, dcm_p in tasks:
        verify_ctvbx(name, ctvbx_p, dcm_p)
