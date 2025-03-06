import numpy as np
import vtk

from openlifu.seg.skinseg import cartesian_to_spherical


def create_plane_actor(center, normal, plane_size=5.0, resolution=5, color=(1.0,0.5,0.5)):
    """
    Create a VTK actor of a plane:
      - The plane is centered at 'center'.
      - Its normal is aligned with 'normal'.
      - 'plane_size' controls how large the drawn plane is (square).
      - 'resolution' controls how many subdivisions in each direction.
      - color is the color of the plane
    """
    # vtkPlaneSource by default creates a plane in the XY-plane (normal=+Z).
    # We'll override that to orient it along 'normal' and center it at 'center'.

    plane_source = vtk.vtkPlaneSource()
    # We define the corners of the plane in its local coordinate space
    # (before orientation via SetNormal):
    half = plane_size / 2.0
    plane_source.SetOrigin(-half, -half, 0)       # bottom-left corner
    plane_source.SetPoint1( half, -half, 0)       # bottom-right corner
    plane_source.SetPoint2(-half,  half, 0)       # top-left corner

    # Increase resolution for a smoother mesh (wireframe or shading)
    plane_source.SetXResolution(resolution)
    plane_source.SetYResolution(resolution)

    # Explicitly set the plane’s center and normal
    plane_source.SetCenter(center)
    plane_source.SetNormal(normal)  # VTK automatically orients it to this normal

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane_source.GetOutputPort())

    plane_actor = vtk.vtkActor()
    plane_actor.SetMapper(mapper)

    plane_actor.GetProperty().SetColor(*color)

    return plane_actor

def create_transducer_cube_actor(transform:np.ndarray=None):
    if transform is None:
        transform = np.eye(4)
    cs = vtk.vtkCubeSource()
    axial_size = 17
    elevational_size = 35
    lateral_size = 50
    cs.SetXLength(lateral_size)
    cs.SetYLength(elevational_size)
    cs.SetZLength(axial_size)
    cs.SetCenter((0,0,-axial_size/2)) # place cube below the x-y-plane to serve as a depiction of the transducer body

    matrix_vtk = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix_vtk.SetElement(i, j, transform[i, j])

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix_vtk)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputConnection(cs.GetOutputPort())

    cm = vtk.vtkPolyDataMapper()
    cm.SetInputConnection(transform_filter.GetOutputPort())
    ca = vtk.vtkActor()
    ca.GetProperty().SetColor((0.2,0.6,0.6))
    ca.SetMapper(cm)
    return ca 


def visualize_3d_volume(vtk_image):
    """
    Visualizes a vtkImageData object as a 3D volume with interactive controls.
    """
    # Create a volume mapper and set the input data
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)

    # Create a volume and set its mapper and properties
    volume = vtk.vtkVolume()
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOn()  # Enable shading for better depth perception
    volume_property.SetAmbient(0.2)
    volume_property.SetDiffuse(0.7)
    volume_property.SetSpecular(0.3)

    # Create a color transfer function (maps voxel values to colors)
    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(0, 0.0, 0.0, 0.0)  # Black at minimum value
    color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)  # White at maximum value

    # Create an opacity transfer function (maps voxel values to transparency)
    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(0, 0.0)   # Fully transparent at minimum value
    opacity_func.AddPoint(50, 1.0)  # Semi-transparent for mid-range values
    opacity_func.AddPoint(255, 1.0)  # Fully opaque at maximum value

    # Attach transfer functions to the volume property
    volume_property.SetColor(color_func)
    volume_property.SetScalarOpacity(opacity_func)

    # Set the mapper and property to the volume
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Create a renderer and add the volume
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background for contrast

    # Create a render window and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)  # Window size

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Add an interactor style for volume exploration
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)

    # Start the visualization
    render_window.Render()
    interactor.Start()

def visualize_polydata(polydata, title="PolyData Visualization",
                       highlight_points=None,
                       highlight_point_vals=None,
                       camera_start=None,
                       camera_focus=None,
                       additional_actors=None,
                       animation_interval_ms=None):
    """
    Visualizes a vtkPolyData object using VTK with enhanced interaction.
    Optionally, highlights a specified point with a yellow sphere.
    If additional_actors are provided, toggles through them one-by-one
    on a timer, hiding all but the current actor.

    Parameters:
        polydata (vtkPolyData or list): The mesh data to visualize.
        title (str): The title of the render window.
        highlight_points (tuple or list of tuples): Points to highlight with spheres.
        highlight_point_vals: optionally a list of floats of the same length as hgihlight poitns to color the points yellow to blue
        camera_start (tuple): Starting camera position (x, y, z).
        camera_focus (tuple): Camera focal point (x, y, z).
        additional_actors (list): Actors to be possibly animated/toggled in sequence.
        animation_interval_ms (int): Interval in milliseconds for stepping through actors.
            (If None then all actors are shown at once with no animation)
    """

    if not isinstance(polydata, list):
        polydata = [polydata]

    # Create mappers and actors for each polydata
    mappers = [vtk.vtkPolyDataMapper() for _ in polydata]
    for mapper, pd in zip(mappers, polydata):
        mapper.SetInputData(pd)

    actors = []
    for i, mapper in enumerate(mappers):
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # Optionally tweak how the main mesh is shown
        actor.GetProperty().SetColor(0.8, 0.8, (i + 1) / len(mappers))
        actor.GetProperty().SetEdgeVisibility(0)
        actor.GetProperty().SetEdgeColor(0, 0, 0)
        actors.append(actor)

    # Create a renderer and add the main actors
    renderer = vtk.vtkRenderer()
    for actor in actors:
        renderer.AddActor(actor)

    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark gray

    # Reset camera so that it frames the main polydata
    renderer.ResetCamera()
    
    # If highlight_points given, add a colored sphere for each
    if highlight_points is not None:        
        if not isinstance(highlight_points, list):
            highlight_points = [highlight_points]

        if highlight_point_vals is None:
            highlight_point_vals = [1.] * len(highlight_points)
        if len(highlight_point_vals) != len(highlight_points): raise ValueError()
        vmin = min(highlight_point_vals)
        vmax = max(highlight_point_vals)

        for point,val in zip(highlight_points,highlight_point_vals):
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(*point)
            sphere_source.SetRadius(0.5)
            sphere_source.SetThetaResolution(5)
            sphere_source.SetPhiResolution(5)

            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)

            t = (val - vmin) / (vmax - vmin) if vmax > vmin else 0. # color interpolation parameter 0 to 1
            color = (1-t)*np.array([1.0,1.0,0.0]) + t*np.array([0.0,0.0,1.0]) # yellow to blue
            sphere_actor.GetProperty().SetColor(*color)

            renderer.AddActor(sphere_actor)

    # Create a render window and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(title)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Use a trackball style
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)

    # Orientation axes in a corner
    axes = vtk.vtkAxesActor()
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(interactor)
    axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    axes_widget.SetEnabled(1)
    axes_widget.InteractiveOn()

    # Optionally define camera position/focus
    camera = renderer.GetActiveCamera()
    if camera_start is not None:
        camera.SetPosition(*camera_start)
        camera.SetViewUp(0, 1, 0)
    if camera_focus is not None:
        camera.SetFocalPoint(*camera_focus)
        camera.SetViewUp(0, 1, 0)

    if animation_interval_ms is not None:
        # If you have additional actors that should be animated
        # we'll keep them all in the scene but manage their visibility.
        if additional_actors is not None and len(additional_actors) > 0:
            # Add them all to the renderer but hide them
            for a in additional_actors:
                a.SetVisibility(False)
                renderer.AddActor(a)

            # Show the first actor initially
            additional_actors[0].SetVisibility(True)
            current_index = 0

            def timer_callback(obj, event):
                nonlocal current_index
                # Hide current
                additional_actors[current_index].SetVisibility(False)
                # Move to next
                current_index = (current_index + 1) % len(additional_actors)
                # Show next
                additional_actors[current_index].SetVisibility(True)
                render_window.Render()

            interactor.AddObserver("TimerEvent", timer_callback)
            interactor.CreateRepeatingTimer(animation_interval_ms)
    else:
        if additional_actors is not None:
            for actor in additional_actors:
                renderer.AddActor(actor)

    # Render and start interaction
    render_window.Render()
    interactor.Initialize()
    interactor.Start()


def sphere_from_interpolator(interpolator, theta_res:int = 50, phi_res:int = 50) -> vtk.vtkPolyData:
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(1.0)
    sphere_source.SetThetaResolution(theta_res)  # Set the resolution in the theta direction
    sphere_source.SetPhiResolution(phi_res)  # Set the resolution in the phi direction
    sphere_source.Update()
    sphere_polydata = sphere_source.GetOutput()
    sphere_points = sphere_polydata.GetPoints()
    for i in range(sphere_points.GetNumberOfPoints()):
        point = np.array(sphere_points.GetPoint(i))
        r, theta, phi = cartesian_to_spherical(*point)
        r = interpolator(theta, phi)
        sphere_points.SetPoint(i, r * point)
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(sphere_polydata)
    normals_filter.Update()
    return normals_filter.GetOutput()
