import struct
import json
import numpy as np
import os

class CtvbxReader:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.depth = 0
        self.voxel_size = 1.0
        self.is_signed = False
        self.project_data = {}
        self.data = None

    def load(self, file_path):
        """
        Loads a .ctvbx file and returns the volume data as a numpy array.
        """
        with open(file_path, 'rb') as f:
            # 1. Magic (6 bytes)
            magic = f.read(6).decode('utf-8')
            if magic != "CTVIEW":
                raise ValueError("Not a valid CTVBX file.")

            # 2. Version (int32)
            version = struct.unpack('<i', f.read(4))[0]
            if version < 3:
                # Handle legacy versions if necessary, but focusing on v3
                pass

            # 3. Dimensions & Geometry (v2+)
            if version >= 2:
                self.width = struct.unpack('<i', f.read(4))[0]
                self.height = struct.unpack('<i', f.read(4))[0]
                self.depth = struct.unpack('<i', f.read(4))[0]
                self.voxel_size = struct.unpack('<d', f.read(8))[0]
            
            # 4. IsSigned (v3+)
            if version >= 3:
                # BinaryWriter.Write(bool) in C# writes 1 byte
                self.is_signed = struct.unpack('?', f.read(1))[0]

            # 5. JSON Header (int32 length + bytes)
            json_len = struct.unpack('<i', f.read(4))[0]
            json_bytes = f.read(json_len)
            self.project_data = json.loads(json_bytes.decode('utf-8'))

            # Fallback for old versions where dimensions were in ROI
            if version < 2:
                self.width = self.project_data.get('RoiX2', 0) - self.project_data.get('RoiX1', 0)
                self.height = self.project_data.get('RoiY2', 0) - self.project_data.get('RoiY1', 0)
                self.depth = self.project_data.get('RoiZ2', 0) - self.project_data.get('RoiZ1', 0)
                zoom = self.project_data.get('AxialZoom', 1.0)
                self.voxel_size = 1.0 / zoom if zoom > 0 else 1.0
            
            if version < 3:
                self.is_signed = self.project_data.get('IsSigned', False)

            # 6. Raw Data (16-bit)
            total_voxels = self.width * self.height * self.depth
            dtype = np.int16 if self.is_signed else np.uint16
            
            # Read all remaining data or specific amount
            self.data = np.fromfile(f, dtype=dtype, count=total_voxels)
            
            # Reshape to (Depth, Height, Width) for standard 3D indexing [z, y, x]
            self.data = self.data.reshape((self.depth, self.height, self.width))
            
            return self.data

def load_ctvbx(file_path):
    """Convenience function to load a .ctvbx file."""
    reader = CtvbxReader()
    return reader.load(file_path), reader
