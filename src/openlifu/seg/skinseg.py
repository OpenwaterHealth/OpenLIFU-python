import numpy as np
import skimage.filters
import skimage.measure
import vtk
from scipy.ndimage import distance_transform_edt
from vtk.util.numpy_support import numpy_to_vtk


def take_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Given a boolean image array (or any integer numpy array), return a mask of the largest connected component."""
    mask_labeled = skimage.measure.label(mask)
    connected_component_info = skimage.measure.regionprops(mask_labeled)
    largest_connected_componet_label = connected_component_info[np.argmax([rp.area for rp in connected_component_info])].label
    return (mask_labeled == largest_connected_componet_label)

def compute_foreground_mask(
        vol_array : np.ndarray,
        closing_radius : float = 9.,
        lower_quantile_for_otsu_threshold : float = 0.02,
        upper_quantile_for_otsu_threshold : float = 0.99,
    ) -> np.ndarray:
    """Given a 3D image array, return a boolean mask representing the "foreground."

    Args:
        vol_array: a 3D image array of shape (H,W,D)
        closing_readius: the radius of the ball used in the morphological closing operation
        lower_quantile_for_otsu_threshold: a number from 0 to 1. Before otsu thresholding,
            values below this quantile are omitted from the histogram as outliers.
        upper_quantile_for_otsu_threshold: a number from 0 to 1.  Before otsu thresholding,
            values above this quantile are omitted from the histogram as outliers.

    Returns: a boolean array of shape (H,W,D) representing a foreground mask

    This is essentially a port of the BRAINSTools automated foreground masking algorithm.
    - Original algorithm documentation: https://slicer.readthedocs.io/en/latest/user_guide/modules/brainsroiauto.html
    - Original algorithm code: https://github.com/BRAINSia/BRAINSTools/tree/7c37d9e8c238f66f8a83f997d9c9bb659c494c90/BRAINSROIAuto

    The algorithm roughly works as follows:
    - step 1: otsu thresholding
    - step 2: keep only the largest connected component
    - step 3: morphological closing
    - step 4: hole filling

    The default values of the parameters have been observed to work well for mm-spaced brain MRIs.
    """

    # step 1: otsu-threshold the image to create an initial foreground mask.
    threshold_lower, threshold_upper = np.quantile(
        vol_array,
        [lower_quantile_for_otsu_threshold,upper_quantile_for_otsu_threshold]
    )
    threshold_foreground = skimage.filters.threshold_otsu(
        vol_array[(vol_array >= threshold_lower) & (vol_array <= threshold_upper)]
    )
    foreground_mask = vol_array >= threshold_foreground

    # step 2: keep only the largest connected component to throw out spurious bits.
    foreground_mask = take_largest_connected_component(foreground_mask)

    # step 3: do a morphological closing.
    # while this does fill some holes, that's not the main point since step 4 already fills holes.
    # the point of this step is rather to clean up and smooth out the skin surface of small cavities.
    pad_width = int(closing_radius+2) # pad to avoid the situation where dilation hits the boundary
    foreground_mask_padded = np.pad(foreground_mask, pad_width, mode='constant')
    background_edt = distance_transform_edt(~foreground_mask_padded)
    foreground_dilated = background_edt <= closing_radius
    foreground_dilated_edt = distance_transform_edt(foreground_dilated)
    foreground_closed = foreground_dilated_edt >= closing_radius

    # crop to undo the padding above
    h,w,d = foreground_mask.shape
    p = pad_width
    foreground_mask = foreground_closed[p:p+h,p:p+w,p:p+d]

    # step 4: take the complement of the largest connected component of the current background.
    # the background mask at this point contains the "actual background" and possibly also some
    # holes that are inside the foreground region. the largest connected component of this background
    # mask is considered to be the "actual background."" this step therefore serves to fill
    # any remaining holes in the foreground mask.
    # this step is analogous to the seeded flood fill in the original algorithm:
    # https://github.com/BRAINSia/BRAINSTools/blob/7c37d9e8c238f66f8a83f997d9c9bb659c494c90/BRAINSCommonLib/itkLargestForegroundFilledMaskImageFilter.hxx#L255-L302
    foreground_mask = ~take_largest_connected_component(~foreground_mask)

    return foreground_mask


def vtk_img_from_array_and_affine(vol_array:np.ndarray, affine:np.ndarray) -> vtk.vtkImageData:
    """ Convert a numpy (array, affine) pair into a vtkImageData.

    Args:
        vol_array: a 3D image numpy array with float type data
        affine: a numpy array of shape (4,4) representing the affine matrix of the 3D image.

    Returns: vtkImageData with a copy of vol_array as the underlying image data,
        and with origin, spacing, and direction matrix set according to the affine matrix.

    Since a vtkImageData is intended to represent image data on a structured grid with *orthogonal* axes,
    the upper-left 3x3 submatrix of the affine matrix should be an orthogonal matrix. There will be no error
    if it isn't, since the "direction matrix" of a vtkImageData can be set to be non-orthogonal -- it just isn't
    the intended usage of vtkImageData and could be misinterpreted by downstream vtk filters.

    Maintain a reference to vol_array, so that it is not garbage collected
    (which could leave the vtkImageData pointing to invalid memory -- see vtk.util.numpy_support.numpy_to_vtk documentation).
    """

    matrix_3x3 = affine[:3, :3]
    origin = affine[:3, 3]
    spacing = np.linalg.norm(matrix_3x3, axis=0)
    direction_matrix = matrix_3x3 / spacing[np.newaxis,:]

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(vol_array.shape)
    vtk_img.SetOrigin(origin.tolist())
    vtk_img.SetSpacing(spacing.tolist())

    direction_matrix_vtk = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            direction_matrix_vtk.SetElement(i, j, direction_matrix[i, j])
    vtk_img.SetDirectionMatrix(direction_matrix_vtk)

    vol_array_flat = vol_array.transpose((2,1,0)).ravel(order='C')
    vol_array_vtk = numpy_to_vtk(num_array=vol_array_flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_img.GetPointData().SetScalars(vol_array_vtk)

    return vtk_img

def create_closed_surface_from_labelmap(
    binary_labelmap:vtk.vtkImageData,
    decimation_factor:float=0.,
    smoothing_factor:float=0.5
) -> vtk.vtkPolyData:
    """ Create a surface mesh vtkPolyData from a binary labelmap vtkImageData.

    Args:
        binary_labelmap: input vtkImageData binary labelmap
        decimation_factor: 0.0 for no decimation, 1.0 for maximum reduction.
        smoothing_factor: 0.0 for no smoothing, 1.0 for maximum smoothing.

    Returns:
        vtkPolyData: the resulting surface mesh

    The algorithm here is based on the labelmap-to-closed-surface algorithm in 3D Slicer:
    https://github.com/Slicer/Slicer/blob/677932127c73a6c78654d4afd9458a655a4eef63/Libs/vtkSegmentationCore/vtkBinaryLabelmapToClosedSurfaceConversionRule.cxx#L246-L476
    """

    # step 1: pad by 1 pixel all around with 0s, to ensure that the surface is still closed
    # even if the labelmap runs up against the image boundary.
    padder = vtk.vtkImageConstantPad()
    padder.SetInputData(binary_labelmap)
    extent = binary_labelmap.GetExtent()
    padder.SetOutputWholeExtent(
        extent[0] - 1, extent[1] + 1,
        extent[2] - 1, extent[3] + 1,
        extent[4] - 1, extent[5] + 1,
    )
    padder.Update()
    padded_labelmap = padder.GetOutput()

    # step 1: extract surface
    flying_edges = vtk.vtkDiscreteFlyingEdges3D()
    flying_edges.SetInputData(padded_labelmap)
    flying_edges.ComputeGradientsOff()
    flying_edges.ComputeNormalsOff()
    flying_edges.Update()
    surface_mesh = flying_edges.GetOutput()

    # step 2: decimation
    if decimation_factor > 0.0:
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(surface_mesh)
        decimator.SetFeatureAngle(60)
        decimator.SplittingOff()
        decimator.PreserveTopologyOn()
        decimator.SetMaximumError(1)
        decimator.SetTargetReduction(decimation_factor)
        decimator.Update()
        surface_mesh = decimator.GetOutput()

    # step 3: smoothing
    if smoothing_factor > 0.0:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(surface_mesh)

        # map smoothing factor to passband and iterations, copying the approach taken by Slicer
        passband = pow(10.0, -4.0 * smoothing_factor)
        num_iterations = 20 + int(smoothing_factor * 40)

        smoother.SetNumberOfIterations(num_iterations)
        smoother.SetPassBand(passband)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        surface_mesh = smoother.GetOutput()

    # step 4: compute normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface_mesh)
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    surface_mesh = normals.GetOutput()

    return surface_mesh
