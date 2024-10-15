import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np

class MRIProcessor:
    def read_mri_volume(self, mri_volume_path):
        """Reads the MRI volume from a file."""
        image = sitk.ReadImage(mri_volume_path)
        return image

    def apply_otsu_threshold_global(self, image):
        """Applies Otsu thresholding to the MRI volume and returns the binary image and threshold value."""
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetOutsideValue(0)
        otsu_filter.SetInsideValue(1)
        binary_image = otsu_filter.Execute(image)
        threshold = otsu_filter.GetThreshold()
        return binary_image, threshold
    
    def apply_otsu_threshold_slice_by_slice(self, image):
        """Applies Otsu thresholding slice by slice and adjusts thresholds 
        outside 1 standard deviation from the mean."""
        # Get image dimensions
        image_array = sitk.GetArrayFromImage(image)
        z_dim = image.GetDepth()

        # Initialize list to store thresholds
        thresholds = []

        # Iterate over each slice
        for z in range(z_dim):
            slice_image = sitk.GetImageFromArray(image_array[z, :, :])
            otsu_filter = sitk.OtsuThresholdImageFilter()
            otsu_filter.SetOutsideValue(0)
            otsu_filter.SetInsideValue(1)
            otsu_filter.Execute(slice_image)
            threshold = otsu_filter.GetThreshold()
            thresholds.append(threshold)

        # Calculate mean and standard deviation of thresholds
        mean_threshold = np.mean(thresholds)
        std_threshold = np.std(thresholds)

        # Adjust thresholds outside 1 standard deviation from the mean
        for i, threshold in enumerate(thresholds):
            if abs(threshold - mean_threshold) > std_threshold:
                thresholds[i] = mean_threshold

        # Create a new binary image with adjusted thresholds
        binary_image_array = np.zeros_like(image_array)
        for z in range(z_dim):
            binary_image_array[z, :, :] = image_array[z, :, :] > thresholds[z]
        binary_image = sitk.GetImageFromArray(binary_image_array)
        binary_image.CopyInformation(image)

        return binary_image, thresholds

class VTKConverter:
    def convert_to_vtk_image(self, binary_volume, image_spacing, image_origin, image_direction):
        """Converts a binary volume (NumPy array) to a VTK image."""
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(binary_volume.shape)
        vtk_image.SetSpacing(image_spacing)
        vtk_image.SetOrigin(image_origin)
        vtk_image.SetDirectionMatrix(image_direction)

        flat_binary_volume = binary_volume.flatten(order='F')
        vtk_array = numpy_support.numpy_to_vtk(num_array=flat_binary_volume, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_image.GetPointData().SetScalars(vtk_array)

        return vtk_image

    def extract_surface_mesh(self, vtk_image):
        """Extracts the surface mesh from a VTK image using the marching cubes algorithm."""
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(vtk_image)
        marching_cubes.SetValue(0, 0.5)  # Assuming binary volume with 0 and 1 values
        marching_cubes.Update()
        return marching_cubes.GetOutput()
    
    def smooth_mesh_laplacian(self, mesh, iterations=15):
        """Smooths the surface mesh using Laplacian smoothing."""
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(mesh)
        smoother.SetNumberOfIterations(iterations)
        smoother.Update()
        return smoother.GetOutput()

class SurfaceMeshExtractor:
    def __init__(self):
        self.processor = MRIProcessor()
        self.converter = VTKConverter()

    def extract_surface_mesh_with_otsu(self, mri_volume_path, otus_by_slice=False):
        """Extracts the surface mesh from an MRI volume using Otsu thresholding."""
        # Step 1: Read the MRI volume
        image = self.processor.read_mri_volume(mri_volume_path)

        # Step 2: Apply Otsu thresholding
        if otus_by_slice:
            binary_image, threshold = self.processor.apply_otsu_threshold_slice_by_slice(image)
        else:
            binary_image, threshold = self.processor.apply_otsu_threshold_global(image)

        # Convert the binary image to a NumPy array
        binary_volume = sitk.GetArrayFromImage(binary_image).astype(np.uint8)

        # Step 3: Convert to VTK image
        vtk_image = self.converter.convert_to_vtk_image(binary_volume, image.GetSpacing(), image.GetOrigin(), image.GetDirection())

        # Step 4: Extract the surface mesh
        mesh = self.converter.extract_surface_mesh(vtk_image)

        # Step 5: Smooth the surface mesh
        mesh_smooth = self.converter.smooth_mesh_laplacian(mesh)

        return mesh, mesh_smooth, binary_image, vtk_image, threshold
    
if __name__ == "__main__":
    extractor = SurfaceMeshExtractor()
    mri_volume_path = "/Users/kedar/Desktop/Kiri/IXI-T1/IXI211-HH-1568-T1.nii.gz"  # Replace with your MRI volume path

    # Test with global Otsu thresholding
    mesh_global, mesh_global_smooth, binary_image_global, vtk_image_global, threshold_global = extractor.extract_surface_mesh_with_otsu(mri_volume_path)
    print(f"Global Otsu threshold: {threshold_global}")

    # Test with slice-by-slice Otsu thresholding
    mesh_slice, mesh_slice_smooth, binary_image_slice, vtk_image_slice, thresholds_slice = extractor.extract_surface_mesh_with_otsu(mri_volume_path, otus_by_slice=True)
    print(f"Slice-by-slice Otsu thresholds: {thresholds_slice}")

    # Save the mesh in STL format:
    def save_mesh(mesh, filename):
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Write()

    # Save the global Otsu threshold mesh
    save_mesh(mesh_global, "mesh_global.stl")

    # Save the smoothed global Otsu threshold mesh
    save_mesh(mesh_global_smooth, "mesh_global_smooth.stl")

    # Save the slice-by-slice Otsu threshold mesh
    save_mesh(mesh_slice, "mesh_slice.stl")

    # Save the smoothed slice-by-slice Otsu threshold mesh
    save_mesh(mesh_slice_smooth, "mesh_slice_smooth.stl")
