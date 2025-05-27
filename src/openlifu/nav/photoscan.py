from __future__ import annotations

import datetime
import importlib
import json
import logging
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, CalledProcessError, CompletedProcess, Popen
from typing import Annotated, Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import OpenEXR
import requests
import trimesh
import vtk
from PIL import Image
from vtk.util.numpy_support import numpy_to_vtk

from openlifu.util.annotations import OpenLIFUFieldData

logger_meshrecon = logging.getLogger("MeshRecon")
logger_meshroom = logging.getLogger("Meshroom")

@dataclass
class Photoscan:
    id: Annotated[str, OpenLIFUFieldData("Photoscan ID", "ID of this photoscan")] = "photoscan"
    """ID of this photoscan"""

    name: Annotated[str, OpenLIFUFieldData("Photoscan name", "Photoscan name")] = "Photoscan"
    """Photoscan name"""

    model_filename: Annotated[str | None, OpenLIFUFieldData("Model filename", "Relative path to model")] = None
    """Relative path to model"""

    texture_filename: Annotated[str | None, OpenLIFUFieldData("Texture filename", "Relative path to texture image")] = None
    """Relative path to texture image"""

    mtl_filename: Annotated[str | None, OpenLIFUFieldData("Material filename", "Relative path to materials file")] = None
    """Relative path to materials file"""

    photoscan_approved: Annotated[bool, OpenLIFUFieldData("Approved?", "Approval state of the photoscan. 'True' means the user has provided some kind of confirmation that the photoscan is good enough to be used.")] = False
    """Approval state of the photoscan. 'True' means the user has provided some kind of
    confirmation that the photoscan is good enough to be used."""

    @staticmethod
    def from_json(json_string: str) -> Photoscan:
        """Load a Photoscan from a json string"""
        return Photoscan.from_dict(json.loads(json_string))

    def to_json(self, compact: bool) -> str:
        """Serialize a Photoscan to a json string. This is different to the format written to file
        and does not contain the loaded models and texture.
        Args:
            compact:if enabled then the string is compact (not pretty). Disable for pretty.
        Returns: A json string representing the complete Photoscan object
        """
        from openlifu.util.json import PYFUSEncoder

        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'), cls=PYFUSEncoder)
        else:
            return json.dumps(self.to_dict(), indent=4, cls=PYFUSEncoder)

    @staticmethod
    def from_dict(d:Dict) -> Photoscan:
        """
        Create a Photoscan from a dictionary
        param d: Dictionary of photoscan parameters.
        returns: Photoscan object
        """
        return Photoscan(**d)

    def to_dict(self) -> Dict:
        """
        Convert the photoscan to a dictionary
        returns: Dictionary of photoscan parameters
        """
        d = self.__dict__.copy()
        return d

    @staticmethod
    def from_file(filename) -> Photoscan:
        """
        Load a Photoscan from a metadata file
        :param filename: Name of the file
        """
        with open(filename) as f:
            return Photoscan.from_dict(json.load(f))

    def to_file(self, filename):
        """
        Save the photoscan to a file.
        :param filename: Name of the file
        """
        Path(filename).parent.parent.mkdir(exist_ok=True) #photoscan directory
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact = False))

def load_data_from_filepaths(model_abspath: str, texture_abspath: str) -> Tuple[vtk.vtkPolyData, vtk.vtkImageData]:
    """
    This function returns the data directly from the model and texture filepaths without requiring a photoscan object.
    param model_abspath: absolute filepath to model data
          texture abspath: absolute filepath to texture data
    Returns: Photoscan data as (model_vtkpolydata, texture_vtkimagedata)
    """
    model_polydata  = load_model(model_abspath)
    texture_imagedata = load_texture(texture_abspath)

    return (model_polydata, texture_imagedata)

def load_data_from_photoscan(photoscan: Photoscan, parent_dir) -> Tuple[vtk.vtkPolyData, vtk.vtkImageData]:
    """
    param parent_dir: parent directory containing model and texture data filepaths
    Returns: Photoscan data as (model_vtkpolydata, texture_vtkimagedata)
    """
    model_polydata  = load_model(Path(parent_dir)/photoscan.model_filename)
    texture_imagedata = load_texture(Path(parent_dir)/photoscan.texture_filename)

    return (model_polydata, texture_imagedata)

def load_model(file_name) -> vtk.vtkPolyData:
    """
    This function assumes that the model is saved to file in LPS coordinates.
    The model that is returned is in RAS coordinates
    """
    if not Path(file_name).exists():
        raise FileNotFoundError(f'Model filepath does not exist: {file_name}')
    mesh = read_as_vtkpolydata(file_name)
    mesh_ras = convert_between_ras_and_lps(mesh)

    return mesh_ras

def load_texture(file_name) -> vtk.vtkImageData:
    if not Path(file_name).exists():
        raise FileNotFoundError(f'Texture data filepath does not exist: {file_name}')
    texture = read_as_vtkimagedata(file_name)
    return texture

def read_as_vtkpolydata(file_name):

    suffix_to_reader_dict = {
        '.ply' : vtk.vtkPLYReader,
        '.vtp' : vtk.vtkXMLPolyDataReader,
        '.obj' : vtk.vtkOBJReader,
        '.stl' : vtk.vtkSTLReader,
        '.vtk' : vtk.vtkPolyDataReader,
        '.g' : vtk.vtkBYUReader
    }
    path = Path(file_name)
    ext = path.suffix.lower()
    if path.suffix not in suffix_to_reader_dict:
        raise ValueError(f"File format {ext} not supported by reader")
    reader = suffix_to_reader_dict[ext]()
    if ext == '.g':
        reader.SetGeometryName(file_name)
    else:
        reader.SetFileName(file_name)
    reader.Update()
    poly_data = reader.GetOutput()

    return poly_data

def convert_numpy_to_vtkimage(image_numpy: np.ndarray) -> vtk.vtkImageData:
    """
    Converts a numpy array with dimensions [HxWx3] representing an RGB image into vtkImageData
    """
    vtkimage_data = vtk.vtkImageData()
    vtkimage_data.SetDimensions(image_numpy.shape[1], image_numpy.shape[0], 1)
    vtkimage_data.SetNumberOfScalarComponents(image_numpy.shape[2], vtkimage_data.GetInformation())
    pd = vtkimage_data.GetPointData()
    new_rgb_data = image_numpy.reshape((-1, image_numpy.shape[2]))
    vtk_array = numpy_to_vtk(new_rgb_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
    pd.SetScalars(vtk_array)
    return vtkimage_data

def read_as_vtkimagedata(file_name) -> vtk.vtkImageData:

    suffix_to_reader_dict = {
        '.jpg' : vtk.vtkJPEGReader,
        '.png' : vtk.vtkPNGReader,
        '.tiff' : vtk.vtkTIFFReader,
    }
    path = Path(file_name)
    ext = path.suffix.lower()
    if ext == '.exr':
        with OpenEXR.File(str(file_name), separate_channels = True) as exr_file:
            R = exr_file.channels()['R'].pixels
            G = exr_file.channels()['G'].pixels
            B = exr_file.channels()['B'].pixels
            # Combine channels into a single RGB image (H x W x 3)
            rgb_data = np.stack([R, G, B], axis=-1, dtype = np.float32)
            # EXR stores pixel data using half 16-bit floating point values. Convert data to an 8-bit image compatibility with VTK
            rgb_data = np.clip(rgb_data*(2**16-1), 0, 65535)
            image_data = convert_numpy_to_vtkimage(rgb_data)
    elif ext in suffix_to_reader_dict:
        reader = suffix_to_reader_dict[ext]()
        reader.SetFileName(file_name)
        reader.Update()
        image_data = reader.GetOutput()
    else:
        raise ValueError(f"File format {path.suffix} not supported by reader")

    return image_data

def convert_between_ras_and_lps(mesh : vtk.vtkPointSet) -> vtk.vtkPointSet:
    """Converts a mesh (polydata, unstructured grid or even just a point cloud) between the LPS (left-posterior-superior) coordinate system and
      RAS (right-anterior-superior) coordinate system."""

    transform_ras_to_lps = vtk.vtkTransform()
    transform_ras_to_lps.Scale(-1,-1,1)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(transform_ras_to_lps)
    transformFilter.Update()

    return transformFilter.GetOutput()

def get_meshroom_pipeline_names() -> list[str]:
    """Get a list of names of valid meshroom pipelines that can be used in run_reconstruction"""
    pipeline_dir = importlib.resources.files("openlifu.nav.meshroom_pipelines")
    return [f.stem for f in pipeline_dir.iterdir() if f.suffix == ".mg"]

def subprocess_stream_output(
    args: str | List[str],
    stdout_handler: Callable[[str], None],
    stderr_handler: Callable[[str], None],
    check: bool = True,
    text: bool = True,
    **kwargs: Any,
) -> CompletedProcess:
    """
    Run a subprocess and stream its stdout and stderr output to separate handlers/loggers. A drop in replacement
    for subprocess.run that handles logging.

    Args:
        args (Union[str, List[str]]): Command and arguments to execute. Can be a string or list of strings.
        stdout_handler (Callable[[str], None]): Function to handle each line of standard output.
        stderr_handler (Callable[[str], None]): Function to handle each line of standard error.
        check (bool, optional): If True, raise CalledProcessError if the subprocess exits with a non-zero code.
        text (bool, optional): If True, communicate with the process using text mode. Defaults to True.
        **kwargs (Any): Additional keyword arguments passed to subprocess.Popen.

    Returns:
        CompletedProcess: An object containing the arguments used and the return code.

    Raises:
        CalledProcessError: If `check` is True and the subprocess exits with a non-zero status.
    """
    with Popen(args, stdout=PIPE, stderr=PIPE, text=text, **kwargs) as process:

        def log_stream(stream, handler):
            for line in stream:
                handler(line.rstrip())

        # Create threads for each stream
        stdout_thread = threading.Thread(target=log_stream, args=(process.stdout, stdout_handler))
        stderr_thread = threading.Thread(target=log_stream, args=(process.stderr, stderr_handler))
        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        retcode = process.wait()

    if check and retcode:
        raise CalledProcessError(retcode, process.args)
    return CompletedProcess(process.args, retcode)

#Just hard code node order
_nodes = [  "FeatureExtraction_1",
            "FeatureMatching_1",
            "StructureFromMotion_1",
            "PrepareDenseScene_1",
            "DepthMap_1",
            "DepthMapFilter_1",
            "Meshing_1",
            "MeshFiltering_1",
            "Texturing_1",
            "Publish_1" ]

def run_reconstruction(
    images: list[Path],
    pipeline_name: str = "default_pipeline",
    input_resize_width: int = 3024,
    use_masks: bool = True,
    window_radius: int | None = None,
    return_durations: bool = False,
    progress_callback : Callable[[int,str],None] | None = None,
) -> Tuple[Photoscan, Path] | Tuple[Photoscan, Path, Dict[str, float]]:
    """Run Meshroom with the given images and pipeline.
    Args:
        images (list[Path]): List of image file paths.
        pipeline_name (str): Name of the Meshroom pipeline in meshroom_pipelines folder.
            See also `get_meshroom_pipeline_names`.
        input_resize_width (int): Width to which input images will be resized, in pixels.
        use_masks (bool): Whether to include a background removal step to filter the dense reconstruction.
        window_radius (Optional[int]): number of images forward and backward in the sequence to try and
            match with, if None match each images to all others.
        return_durations (bool): If True, also return a dictionary mapping node names to durations in seconds.
        progress_callback: An optional function that will be called to report progress. The function should accept two arguments:
            an integer progress value from 0 to 100 followed by a string message describing the step currently being worked on.

    Returns:
        Union[Tuple[Photoscan, Path], Tuple[Photoscan, Path, Dict[str, float]]]:
            - If return_durations is False: returns the Photoscan and the data directory path.
            - If return_durations is True: also returns a dictionary of node execution times.
    """

    if progress_callback is None:
        def progress_callback(progress_percent : int, step_description : str): # noqa: ARG001
            pass # Define it to be a no-op if no progress_callback was provided.

    progress_callback(0, "Starting reconstruction")

    durations = {}

    pipeline_dir = importlib.resources.files("openlifu.nav.meshroom_pipelines")
    valid_configs = get_meshroom_pipeline_names()

    if pipeline_name not in valid_configs:
        raise ValueError(
            f"Invalid pipeline name '{pipeline_name}'. "
            f"Valid options are: {', '.join(valid_configs)}"
        )

    pipeline = pipeline_dir / f"{pipeline_name}.mg"

    config_nodes, downscale_factor = get_meshroom_config_info(pipeline)

    if shutil.which("meshroom_batch") is None:
        raise FileNotFoundError("Error: 'meshroom_batch' is not found in system PATH. Ensure it is installed and accessible.")

    temp_dir = Path(tempfile.mkdtemp())

    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    #resize the images and store in tmp dir
    progress_callback(0, "Resizing images")
    start_time = time.perf_counter()
    new_paths = []
    for image in images:
        img = Image.open(image)
        orientation = img.getexif().get(274, 1)
        if orientation in range(4,8):
            #camera is rotated. Get the perceived width.
            old_width = img.height
        else:
            old_width = img.width
        scale = input_resize_width/old_width

        resize_width = int(scale*img.width)
        resize_height = int(scale*img.height)
        min_allowed = downscale_factor * 128
        min_side = min(resize_height, resize_width)

        if downscale_factor > 0 and min_side < min_allowed:
            raise ValueError(
                f"Minimum resized side length is {min_side}, but the minimum allowed for pipeline '{pipeline_name}' is {min_allowed}. "
                "Please increase `input_resize_width` or use a pipeline with less downscaling."
            )

        img = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        img.save( images_dir / image.name, exif=img.getexif())
        new_paths.append(images_dir / image.name)
    durations["ImageResizing"] = time.perf_counter() - start_time

    output_dir = temp_dir / "output"
    cache_dir = temp_dir / "cache"
    pair_file_path = cache_dir / "imageMatches.txt"

    command = [
        "meshroom_batch",
        "--pipeline", pipeline.as_posix(),
        "--output", output_dir.as_posix(),
        "--input", images_dir.as_posix(),
        "--cache", cache_dir.as_posix(),
        "--paramOverrides", f"FeatureMatching_1.imagePairsList={pair_file_path.as_posix()}",
    ]

    if use_masks:
        progress_callback(3, "Generating image masks")
        start_time = time.perf_counter()
        masks_dir = temp_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        make_masks(new_paths, masks_dir)
        command.append( f"PrepareDenseScene_1.masksFolders=['{masks_dir.as_posix()}']" )
        durations["MaskCreation"] = time.perf_counter() - start_time

    #run CameraInit node to set view ids
    progress_callback(8, "Initializing camera IDs")
    command_camera_init = [*command.copy(), "--toNode", "CameraInit"]
    subprocess_stream_output(command_camera_init, logger_meshroom.info, logger_meshroom.warning)

    camera_init_path = next(cache_dir.glob("CameraInit/*/cameraInit.sfm"))
    write_pair_file(new_paths, camera_init_path, pair_file_path, window_radius=window_radius)

    number_of_nodes = len([node for node in _nodes if node in config_nodes])
    pipeline_progress_start = 10.0
    pipeline_progress_end = 90.0
    pipeline_progress_step = (pipeline_progress_end - pipeline_progress_start) / number_of_nodes

    #run each node
    pipeline_progress = pipeline_progress_start
    for node in _nodes:
        if node not in config_nodes:
            continue
        command_i = [*command.copy(), "--toNode", node]

        progress_callback(int(pipeline_progress),f"Meshroom: {node.strip('_1')}")
        start_time = time.perf_counter()
        subprocess_stream_output(command_i, logger_meshroom.info, logger_meshroom.warning)
        durations[node] = time.perf_counter() - start_time
        pipeline_progress += pipeline_progress_step

    progress_callback(pipeline_progress_end, "Merging texture maps")

    #merge texturemaps
    output_dir_merged = temp_dir / "output_dir_merged"
    output_dir_merged.mkdir(parents=True, exist_ok=True)

    merge_textures(output_dir / "texturedMesh.obj", output_dir_merged)

    photoscan_dict = {
        "model_filename":  "texturedMesh.obj",
        "texture_filename": "material_0.png",
        "mtl_filename": "material.mtl",
    }

    photoscan = Photoscan.from_dict(photoscan_dict)
    progress_callback(100, "Complete")
    if return_durations:
        return photoscan, output_dir_merged, durations
    return photoscan, output_dir_merged

def udim_to_tile(udim_str: str) -> Tuple[int, int]:
    """Convert UDIM string to tile coordinates"""
    x = int(udim_str[-2:]) - 1
    tile_u = x % 10
    tile_v = x // 10
    return tile_u, tile_v

def merge_textures(input_obj_path: Path, output_path: Path) -> None:
    """Merge the UDIM textures output by meshroom into a single large texture

    Args:
        input_obj_path (Path): Path to .obj mesh to merge.
        output_path (Path): Path to save merged .obj mesh.
    """
    scene = trimesh.load(input_obj_path, process=True)

    if isinstance(scene, trimesh.Scene):
        mesh_dict = scene.geometry
    else:
        mesh_dict = {'material_1001': scene}

    for name, mesh in mesh_dict.items():
        mesh.tile = udim_to_tile(name[-4:])

    num_u_tiles = max(mesh.tile[0] for _, mesh in mesh_dict.items()) + 1
    num_v_tiles = max(mesh.tile[1] for _, mesh in mesh_dict.items()) + 1

    first_tile = np.array(mesh_dict['material_1001'].visual.material.image)
    tile_height, tile_width, tile_ch = first_tile.shape
    tex_height, tex_width = num_v_tiles*tile_height, num_u_tiles*tile_height
    new_texture = np.zeros((tex_height, tex_width, tile_ch), dtype=first_tile.dtype)

    new_verts = []
    new_faces = []
    new_uvs = []
    num_verts = 0
    for _, mesh in mesh_dict.items():
        tile_u, tile_v = mesh.tile
        texture_tile = np.array(mesh.visual.material.image)
        new_texture[tex_height - tile_height*(tile_v+1): tex_height-tile_height*tile_v, tile_width*tile_u:tile_width*(tile_u+1)] = texture_tile
        new_verts.append(mesh.vertices)
        new_faces.append(mesh.faces+num_verts)
        new_uvs.append(mesh.visual.uv)
        num_verts += mesh.vertices.shape[0]

    new_texture = Image.fromarray(new_texture)
    new_verts = np.concatenate(new_verts, axis=0)
    new_faces = np.concatenate(new_faces, axis=0)
    new_uvs = np.concatenate(new_uvs, axis=0)

    new_uvs = new_uvs/np.array([num_u_tiles, num_v_tiles]).reshape(-1,2)

    mesh = trimesh.Trimesh() #Note putting vertices in this constructor will merge nearby ones
    mesh.vertices = new_verts
    mesh.faces = new_faces
    mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, image=new_texture)

    mesh.export(output_path / "texturedMesh.obj")


def apply_exif_orientation_numpy(image: np.ndarray, orientation: int, inverse: bool = False) -> np.ndarray:
    """
    Transforms an image array to undo or redo EXIF orientation.

    Args:
        image (np.ndarray): The image array (H x W x C or H x W).
        orientation (int): EXIF orientation tag value (1-8).
        inverse (bool): If True, applies the inverse transformation.

    Returns:
        np.ndarray: The transformed image.
    """
    if orientation <= 1 or orientation > 8:
        return image  # No transformation needed

    # Define forward transformations
    def transform(img):
        if orientation == 2:
            return np.fliplr(img)
        elif orientation == 3:
            return np.rot90(img, 2)
        elif orientation == 4:
            return np.flipud(img)
        elif orientation == 5:
            return np.rot90(np.fliplr(img), 1)
        elif orientation == 6:
            return np.rot90(img, -1)
        elif orientation == 7:
            return np.rot90(np.fliplr(img), -1)
        elif orientation == 8:
            return np.rot90(img, 1)
        else:
            raise ValueError("Unknown exif orientation")

    # Define inverse transformations
    def inverse_transform(img):
        if orientation == 2:
            return np.fliplr(img)
        elif orientation == 3:
            return np.rot90(img, 2)
        elif orientation == 4:
            return np.flipud(img)
        elif orientation == 5:
            return np.fliplr(np.rot90(img, -1))
        elif orientation == 6:
            return np.rot90(img, 1)
        elif orientation == 7:
            return np.fliplr(np.rot90(img, 1))
        elif orientation == 8:
            return np.rot90(img, -1)
        else:
            raise ValueError("Unknown exif orientation")

    return inverse_transform(image) if inverse else transform(image)


def get_modnet_path() -> Path:
    """Get the MODNet checkpoint path. Download it if not present.
    """
    package = "openlifu.nav.modnet_checkpoints"
    filename = "modnet_photographic_portrait_matting.onnx"
    url = "https://data.kitware.com/api/v1/file/67feb2cb31a330568827ab32/download"
    try:
        # Try to find the checkpoint in the package
        resource_path = importlib.resources.files(package) / filename
        if resource_path.is_file():
            logger_meshrecon.info(f"Found existing MODNet checkpoint at {resource_path}")
            return resource_path
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    # Fallback: Download the checkpoint
    base_dir = Path(importlib.resources.files(package))
    full_path = base_dir / filename
    logger_meshrecon.info(f"MODNet checkpoint not found. Downloading from {url}...")
    response = requests.get(url, stream=True, timeout=(10, 300))
    if response.status_code == 200:
        with open(full_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger_meshrecon.info(f"Downloaded MODNet checkpoint to {full_path}")
    else:
        raise RuntimeError(f"Failed to download MODNet checkpoint: {response.status_code} - {response.text}")

    return full_path

def preprocess_image_modnet(image: np.ndarray, ref_size: int = 512) -> np.ndarray:
    """
    Preprocess an input image for MODNet inference.

    This function performs the same preprocessing steps as the official MODNet code:
    - Normalizes image values to the range [-1, 1]
    - Resizes the image based on a reference size (512), maintaining aspect ratio
    - Ensures the resized dimensions are divisible by 32
    - Converts the image to CHW format and adds a batch dimension

    Args:
        image (np.ndarray): Input image in HWC format with values in [0, 255].

    Returns:
        np.ndarray: Preprocessed image in NCHW format with float32 values in [-1, 1].
    """

    # Normalize image to [-1, 1]
    image = image.astype(np.float32) / 255.0
    image = 2 * image - 1

    im_h, im_w, _ = image.shape

    # Resize if both dimensions are smaller or larger than the reference size
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w * ref_size / im_h)
        else:
            im_rw = ref_size
            im_rh = int(im_h * ref_size / im_w)
    else:
        im_rh, im_rw = im_h, im_w

    # Ensure dimensions are divisible by 32
    im_rw -= im_rw % 32
    im_rh -= im_rh % 32

    # Resize, convert to CHW, and add batch dimension
    image = cv2.resize(image, (im_rw, im_rh), interpolation=cv2.INTER_AREA)
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    return image


def make_masks(image_paths: list[Path], output_dir: Path, threshold: float = 0.01) -> None:
    """
    Runs MODNet on a list of image paths and saves the output masks.

    Each output mask is saved in `output_dir` using the original filename with a `.png` extension.
    The `threshold` parameter is used to convert the MODNet soft segmentation output into a binary (hard) mask.
    EXIF orientation data is preserved to ensure correct image alignment when loading into Meshroom.

    Args:
        image_paths (List[str]): List of input image file paths.
        output_dir (str): Directory where the output masks will be saved.
        threshold (float): Threshold to binarize the soft segmentation output.
    """
    # Load the ONNX model
    ckpt_path = get_modnet_path()
    session = ort.InferenceSession(ckpt_path, providers=["CPUExecutionProvider"])  # or CUDAExecutionProvider
    for image_path in image_paths:
        image = Image.open(image_path)
        exif = image.getexif()
        orientation = exif.get(274,1)
        image = np.array(image)
        image = apply_exif_orientation_numpy(image, orientation=orientation)
        im_h, im_w, _ = image.shape
        image = preprocess_image_modnet(image)

        # Run the model
        mask = session.run(["output"], {"input": image})[0].squeeze()
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_AREA)
        mask = mask > threshold
        mask = (mask.astype(np.float32)*255).astype(np.uint8)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        mask = apply_exif_orientation_numpy(mask, orientation=orientation, inverse=True)

        mask = Image.fromarray(mask)
        output_path = output_dir / (image_path.stem + ".png")
        mask.save(output_path, exif=exif)

def _read_path_to_view_ids(input_path: Path) -> Dict[Path, str]:
    """Read the mapping from image paths to viewIds produced by Meshroom CameraInit node"""
    with open(input_path) as f:
        data = json.load(f)

    basename_to_viewid = {
        Path(view['path']): view['viewId']
        for view in data.get('views', [])
    }
    return basename_to_viewid


def _make_pairs(view_ids: List[str], window_radius: int | None = None) -> List[List[str]]:
    """
    Generate image match pairs from a list of view IDs. Assumes view_ids are sequential and
    wrap around. Each view is matched with the `window_radius` views before and after it.
    If `window_radius` is None or large enough to cover all views, exhaustive matching is used.

    Args:
        view_ids (List[str]): Ordered list of view identifiers.
        window_radius (Optional[int]): Number of neighbors to match on each side.

    Returns:
        rows (List[List[str]]): Each row is of the form [src, tgt1, tgt2, ...], meaning src
                                is matched to tgt1, tgt2, etc.
    """
    num_views = len(view_ids)

    if window_radius is None or 2*window_radius + 1 >= num_views:
        rows = [[view_ids[i] for i in range(j, num_views)] for j in range(num_views)]
    else:
        rows = []
        for i in range(num_views):
            rows.append([view_ids[i]])
            for j in range(window_radius):
                k =  i+j+1
                if k < num_views:
                    rows[i].append(view_ids[k])
                else:
                    rows[k % num_views].append(view_ids[i])

    rows = rows[:-1]
    return rows

def write_pair_file(image_paths: List[Path], camera_init_file: Path, output_path: Path, window_radius: int | None=None) -> None:
    """
    Write the imagePairsList file for Meshroom to perform sequential matching with wrap around.
    Assumes that `image_paths` are ordered sequentially and wrap around.
    """
    path_to_viewid = _read_path_to_view_ids(camera_init_file)
    view_ids = [path_to_viewid[im_path] for im_path in image_paths]

    rows = _make_pairs(view_ids, window_radius)

    with open(output_path, "w") as f:
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")


def _get_datetime_taken(image_path: Path) -> datetime.datetime | None:
    """Get the date and time from image metadata."""
    with Image.open(image_path) as img:
        value = img.getexif().get(306, None)
        if value is None:
            return None
        else:
            return datetime.datetime.strptime(value, '%Y:%m:%d %H:%M:%S')

def _extract_numbers(filename: str) -> List[int]:
    """Extracts all sub-strings of numbers as a list of ints."""
    return list(map(int, re.findall(r'\d+', filename)))

def preprocess_image_paths(paths: List[Path],
                           sort_by: str = 'filename',
                           sampling_rate: int = 1) -> List[Path]:
    """
    Sorts and sub-samples a list of image file paths.

    Args:
        paths (List[Path]): A list of image file paths to be processed.
        sort_by (str): Method to sort the images. Options are:
            - 'filename': Sort by numeric values extracted from filenames.
            - 'metadata': Sort by the DateTimeOriginal field in EXIF metadata.
        sampling_rate (int): Return every n-th image after sorting. Must be >= 1.

    Returns:
        List[Path]: A sorted and sub-sampled list of image paths.

    Raises:
        ValueError: If `sort_by` is not one of the valid options.
        ValueError: If `sampling_rate` is less than 1.
        ValueError: If sorting by metadata and no valid EXIF DateTimeOriginal is found.
    """
    if sampling_rate < 1:
        raise ValueError("sampling_rate must be a positive integer.")

    if sort_by == 'metadata':
        dated = []

        for path in paths:
            dt = _get_datetime_taken(path)
            if dt is None:
                raise ValueError(f"Tried to sort by metadata, but no valid 'DateTimeOriginal' metadata was found in image: {path}")
            dated.append((dt, path))

        dated.sort(key=lambda x: x[0])
        paths = [p for _, p in dated]
    elif sort_by == 'filename':
        paths = sorted(paths, key=lambda p: _extract_numbers(p.name))
    else:
        raise ValueError(f"Invalid sort_by value '{sort_by}'. Use 'metadata' or 'filename'.")

    return paths[::sampling_rate]

def get_meshroom_config_info(pipeline_path: str) -> Tuple[List[str], int]:
    """
    Extract the node names and downscale factor used in the DepthMap_1 node from a Meshroom pipeline.

    Args:
        pipeline_path (str): Path to the Meshroom pipeline JSON file.

    Returns:
        Tuple[List[str], int]: A list of node names in the pipeline, and the downscale factor.
            - Returns 2 if the downscale is not explicitly set (Meshroom default).
            - Returns -1 if the DepthMap_1 node is not present.
    """
    with open(pipeline_path) as f:
        pipeline_graph = json.load(f)['graph']

    nodes = list(pipeline_graph.keys())

    downsample_factor = (
        pipeline_graph['DepthMap_1']['inputs'].get('downscale', 2)
        if 'DepthMap_1' in pipeline_graph else -1
    )

    return nodes, downsample_factor
