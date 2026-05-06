import struct
import json
import numpy as np
import os
import datetime

class CtvbxVolume:
    """
    Represents a CT volume loaded from a .ctvbx file.
    Provides easy access to metadata and pixel data.
    """
    def __init__(self, data, header):
        self.data = data  # numpy array [z, y, x]
        self.width = header['width']
        self.height = header['height']
        self.depth = header['depth']
        self.voxel_size = header['voxel_size']
        self.is_signed = header['is_signed']
        self.z_positions = header['z_positions']
        self.project_data = header['project_data']
        self.version = header['version']

    @property
    def window_center(self):
        return self.project_data.get('Wl', 400)

    @property
    def window_width(self):
        return self.project_data.get('Ww', 1500)

    @property
    def threshold(self):
        return self.project_data.get('Threshold', 0)

    def get_parameter(self, key, default=None):
        """Gets a custom project parameter."""
        return self.project_data.get(key, default)

    def export_dicom(self, output_dir, patient_name="Anonymous", patient_id="0001"):
        """
        Exports the volume data to DICOM files.
        Standardizes on Unsigned 16-bit representation for compatibility.
        """
        try:
            import pydicom
            from pydicom.dataset import Dataset, FileDataset
            from pydicom.uid import ExplicitVRLittleEndian, generate_uid
        except ImportError:
            raise ImportError("pydicom and numpy are required for DICOM export.")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        study_instance_uid = generate_uid()
        series_instance_uid = generate_uid()
        now = datetime.datetime.now()
        date_str = now.strftime('%Y%m%d')
        time_str = now.strftime('%H%M%S.%f')

        # Determine intensity mapping (Match C# app logic)
        # If signed (e.g. Toshiba raw), we apply 32768 offset and set Intercept to -32768
        apply_offset = self.is_signed
        intercept = -32768.0 if apply_offset else 0.0
        
        # Display window (WL/WW)
        # Note: If we apply offset, WL must also be offset for correct display in DICOM viewers
        wl = self.window_center + (32768 if apply_offset else 0)
        ww = self.window_width

        for i in range(self.depth):
            slice_raw = self.data[i, :, :]
            
            # Map to Unsigned 16-bit
            if apply_offset:
                slice_data = (slice_raw.astype(np.int32) + 32768).clip(0, 65535).astype(np.uint16)
            else:
                slice_data = np.clip(slice_raw, 0, 65535).astype(np.uint16)

            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.1"

            filename = os.path.join(output_dir, f"IM_{i:04d}.dcm")
            ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

            ds.PatientName = patient_name
            ds.PatientID = patient_id
            ds.ContentDate = date_str
            ds.ContentTime = time_str
            ds.StudyInstanceUID = study_instance_uid
            ds.SeriesInstanceUID = series_instance_uid
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.Modality = "CT"
            
            ds.Rows, ds.Columns = self.height, self.width
            ds.PixelSpacing = [self.voxel_size, self.voxel_size]
            ds.SliceThickness = self.voxel_size
            ds.SpacingBetweenSlices = self.voxel_size
            
            z_pos = self.z_positions[i] if i < len(self.z_positions) else i * self.voxel_size
            ds.ImagePositionPatient = [0, 0, z_pos]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.InstanceNumber = i + 1

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0 # Always Unsigned for consistency
            
            ds.WindowCenter = str(int(wl))
            ds.WindowWidth = str(int(ww))
            ds.RescaleIntercept = str(intercept)
            ds.RescaleSlope = "1"

            ds.PixelData = slice_data.tobytes()
            ds.save_as(filename)

        print(f"Exported {self.depth} slices to {output_dir}")

class CtvbxReader:
    def load(self, file_path):
        """Loads a .ctvbx file (supports v1.x and v2.0) and returns a CtvbxVolume object."""
        with open(file_path, 'rb') as f:
            # 1. Peek for Version 2.0 signature (8 bytes)
            sig_bytes = f.read(8)
            if sig_bytes.startswith(b"CTVBX2.0"):
                return self._load_v2(f, file_path)
            
            # Fallback to v1.x logic
            f.seek(0)
            return self._load_v1(f)

    def _load_v2(self, f, file_path):
        """Internal loader for CTVBX v2.0 format."""
        # Index area is already at offset 8 after reading signature
        json_offset = struct.unpack('<q', f.read(8))[0]
        json_len = struct.unpack('<q', f.read(8))[0]
        binary_offset = struct.unpack('<q', f.read(8))[0]
        binary_len = struct.unpack('<q', f.read(8))[0]

        # Load Metadata
        f.seek(json_offset)
        json_bytes = f.read(json_len)
        project_data = json.loads(json_bytes.decode('utf-8'))

        # Extract dimensions from metadata
        width = project_data.get('VolumeWidth', 0)
        height = project_data.get('VolumeHeight', 0)
        depth = project_data.get('VolumeDepth', 0)
        voxel_size = project_data.get('VoxelSize', 1.0)
        is_signed = project_data.get('IsSigned', False)
        z_positions = project_data.get('ZPositions', [])

        if not z_positions:
            z_positions = [i * voxel_size for i in range(depth)]

        # Load Binary Data
        f.seek(binary_offset)
        total_voxels = width * height * depth
        dtype = np.int16 if is_signed else np.uint16
        
        # Optimized: read exact amount
        data = np.fromfile(f, dtype=dtype, count=total_voxels)
        data = data.reshape((depth, height, width))

        header = {
            'width': width, 'height': height, 'depth': depth,
            'voxel_size': voxel_size, 'is_signed': is_signed,
            'z_positions': z_positions, 'project_data': project_data,
            'version': 2.0
        }
        return CtvbxVolume(data, header)

    def _load_v1(self, f):
        """Internal loader for legacy CTVBX formats."""
        # 1. Magic (6 bytes "CTVIEW")
        magic_bytes = f.read(6)
        magic = magic_bytes.decode('utf-8')
        if magic != "CTVIEW":
            raise ValueError(f"Invalid format signature: {magic}")

        # 2. Version (int32)
        version = struct.unpack('<i', f.read(4))[0]

        # 3. Dimensions & Geometry
        width = struct.unpack('<i', f.read(4))[0]
        height = struct.unpack('<i', f.read(4))[0]
        depth = struct.unpack('<i', f.read(4))[0]
        voxel_size = struct.unpack('<d', f.read(8))[0]
        
        # 4. IsSigned
        is_signed = struct.unpack('?', f.read(1))[0]

        # 5. ZPositions (v4+)
        z_positions = []
        if version >= 4:
            z_len = struct.unpack('<i', f.read(4))[0]
            if z_len > 0:
                z_positions = list(struct.unpack(f'<{z_len}d', f.read(z_len * 8)))
        
        if not z_positions:
            z_positions = [i * voxel_size for i in range(depth)]

        # 6. JSON Header (int32 length + bytes)
        json_len = struct.unpack('<i', f.read(4))[0]
        json_bytes = f.read(json_len)
        project_data = json.loads(json_bytes.decode('utf-8'))

        # 7. Raw Data (16-bit)
        total_voxels = width * height * depth
        dtype = np.int16 if is_signed else np.uint16
        
        data = np.fromfile(f, dtype=dtype, count=total_voxels)
        data = data.reshape((depth, height, width))
        
        header = {
            'width': width, 'height': height, 'depth': depth,
            'voxel_size': voxel_size, 'is_signed': is_signed,
            'z_positions': z_positions, 'project_data': project_data,
            'version': version
        }
        
        return CtvbxVolume(data, header)

def load_volume(file_path):
    """Utility function to load a volume."""
    return CtvbxReader().load(file_path)
