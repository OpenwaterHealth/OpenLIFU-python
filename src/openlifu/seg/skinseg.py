import numpy as np
import skimage.filters
import skimage.measure
from scipy.ndimage import distance_transform_edt


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
