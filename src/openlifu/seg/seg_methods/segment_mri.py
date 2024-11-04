from dataclasses import dataclass

import ants
from nibabel import Nifti1Image

from openlifu.seg.seg_methods.seg_method import SegmentationMethod


@dataclass
class SegmentMRI(SegmentationMethod):
    def _segment(self, volume: Nifti1Image):
        # On using ANTs antropos, see this discussion: https://neurostars.org/t/n-a-whitematter-confounds-in-tsv-output-file/4859/10
        ants_img = ants.from_numpy(
            volume.get_fdata(),
            origin=volume.affine[:3, -1].tolist(),
            spacing=volume.header['pixdim'][:3].tolist(),
            direction=volume.affine[:3, :3]
        )
        mask = ants.get_mask(ants_img)
        img_segs = ants.atropos(ants_img, x=mask, m="[0.2, 1x1x1]")

        return img_segs
