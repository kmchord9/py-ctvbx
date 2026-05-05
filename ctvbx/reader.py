import struct
import json
import numpy as np
import os
import datetime

class CtvbxReader:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.depth = 0
        self.voxel_size = 1.0
        self.is_signed = False
        self.z_positions = []
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

            # 3. Dimensions & Geometry (v2+)
            if version >= 2:
                self.width = struct.unpack('<i', f.read(4))[0]
                self.height = struct.unpack('<i', f.read(4))[0]
                self.depth = struct.unpack('<i', f.read(4))[0]
                self.voxel_size = struct.unpack('<d', f.read(8))[0]
            
            # 4. IsSigned (v3+)
            if version >= 3:
                self.is_signed = struct.unpack('?', f.read(1))[0]

            # 4.5 ZPositions (v4+)
            if version >= 4:
                z_len = struct.unpack('<i', f.read(4))[0]
                if z_len > 0:
                    self.z_positions = list(struct.unpack(f'<{z_len}d', f.read(z_len * 8)))
                else:
                    self.z_positions = []

            # 5. JSON Header (int32 length + bytes)
            json_len = struct.unpack('<i', f.read(4))[0]
            json_bytes = f.read(json_len)
            self.project_data = json.loads(json_bytes.decode('utf-8'))

            # Fallback/Consistency
            if version < 2:
                self.width = self.project_data.get('RoiX2', 0) - self.project_data.get('RoiX1', 0)
                self.height = self.project_data.get('RoiY2', 0) - self.project_data.get('RoiY1', 0)
                self.depth = self.project_data.get('RoiZ2', 0) - self.project_data.get('RoiZ1', 0)
                zoom = self.project_data.get('AxialZoom', 1.0)
                self.voxel_size = 1.0 / zoom if zoom > 0 else 1.0
            
            if version < 3:
                self.is_signed = self.project_data.get('IsSigned', False)
            
            if not self.z_positions:
                # Default to equal spacing if not available
                self.z_positions = [i * self.voxel_size for i in range(self.depth)]

            # 6. Raw Data (16-bit)
            total_voxels = self.width * self.height * self.depth
            dtype = np.int16 if self.is_signed else np.uint16
            
            self.data = np.fromfile(f, dtype=dtype, count=total_voxels)
            self.data = self.data.reshape((self.depth, self.height, self.width))
            
            return self.data

    def export_dicom(self, output_dir, patient_name="Anonymous", patient_id="0001"):
        """
        Exports the volume data to a set of DICOM files.
        Requires pydicom.
        """
        try:
            import pydicom
            from pydicom.dataset import Dataset, FileDataset
            from pydicom.uid import ExplicitVRLittleEndian, generate_uid
        except ImportError:
            raise ImportError("pydicom is required for DICOM export. Install it with 'pip install pydicom'.")

        if self.data is None:
            raise ValueError("No data loaded to export.")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Common metadata
        study_instance_uid = generate_uid()
        series_instance_uid = generate_uid()
        now = datetime.datetime.now()
        date_str = now.strftime('%Y%m%d')
        time_str = now.strftime('%H%M%S.%f')

        for i in range(self.depth):
            slice_data = self.data[i, :, :]
            
            # File metadata
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2' # CT Image Storage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = "1.2.3.4"
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            filename = os.path.join(output_dir, f"slice_{i:04d}.dcm")
            ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

            # Patient/Study Info
            ds.PatientName = patient_name
            ds.PatientID = patient_id
            ds.ContentDate = date_str
            ds.ContentTime = time_str
            ds.StudyInstanceUID = study_instance_uid
            ds.SeriesInstanceUID = series_instance_uid
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.Modality = "CT"
            
            # Geometry
            ds.Rows, ds.Columns = self.height, self.width
            ds.PixelSpacing = [self.voxel_size, self.voxel_size]
            ds.SliceThickness = self.voxel_size
            
            z_pos = self.z_positions[i] if i < len(self.z_positions) else i * self.voxel_size
            ds.ImagePositionPatient = [0, 0, z_pos]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0] # Axial
            ds.InstanceNumber = i + 1

            # Pixel Data Info
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1 if self.is_signed else 0
            
            # Windowing (optional but helpful)
            ds.WindowCenter = str(self.project_data.get('Wl', 40))
            ds.WindowWidth = str(self.project_data.get('Ww', 400))
            ds.RescaleIntercept = "0"
            ds.RescaleSlope = "1"

            # Set pixel data
            ds.PixelData = slice_data.tobytes()

            ds.save_as(filename)

        print(f"Exported {self.depth} slices to {output_dir}")

def load_ctvbx(file_path):
    reader = CtvbxReader()
    return reader.load(file_path), reader
