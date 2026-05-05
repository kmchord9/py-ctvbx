# py-ctvbx
Python library for reading .ctvbx (CT Viewer Binary eXtended) files.

## Installation
```bash
pip install .
```

## Usage
```python
import ctvbx
import matplotlib.pyplot as plt

# Load volume data and metadata
data, info = ctvbx.load_ctvbx("sample.ctvbx")

print(f"Dimensions: {info.width}x{info.height}x{info.depth}")
print(f"Voxel Size: {info.voxel_size} mm")
print(f"Is Signed: {info.is_signed}")

# Display a slice (Axial)
plt.imshow(data[info.depth // 2, :, :], cmap='gray')
plt.show()

# Export to DICOM for verification
info.export_dicom("exported_dicom", patient_name="Test Patient")
```
