import numpy as np
import vtk

from openlifu.seg.skinseg import cartesian_to_spherical


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

def visualize_polydata(polydata, title="PolyData Visualization", highlight_points=None, camera_start=None, camera_focus=None):
    """
    Visualizes a vtkPolyData object using VTK with enhanced interaction.
    Optionally, highlights a specified point with a yellow sphere.

    Parameters:
        polydata (vtkPolyData): The mesh data to visualize. Could give a list of meshes if you want multiple.
        title (str): The title of the render window.
        highlight_point (tuple): A tuple of (x, y, z) coordinates to highlight with a yellow sphere.
    """
    if not isinstance(polydata, list):
        polydata = [polydata]

    # Create mappers for the polydatas
    mappers = [vtk.vtkPolyDataMapper() for _ in polydata]
    for mapper,pd in zip(mappers,polydata):
        mapper.SetInputData(pd)

    # Create an actor for the polydata
    actors = [vtk.vtkActor() for _ in mappers]
    for i,(actor,mapper) in enumerate(zip(actors, mappers)):
        actor.SetMapper(mapper)

        # Optional: Add properties to the actor for better visualization
        actor.GetProperty().SetColor(0.8, 0.8, (i+1)/len(actors))
        actor.GetProperty().SetEdgeVisibility(0)     # Turn on to show mesh edges
        actor.GetProperty().SetEdgeColor(0, 0, 0)    # Black edges

    # Create a renderer
    renderer = vtk.vtkRenderer()
    for actor in actors:
        renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark gray background

    # Automatically adjust the camera to fit the polydata
    renderer.ResetCamera()

    # If a highlight point is provided, create and add a yellow sphere
    if highlight_points is not None:
        if not isinstance(highlight_points,list):
            highlight_points = [highlight_points]
        for highlight_point in highlight_points:
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(*highlight_point)
            sphere_source.SetRadius(0.5)
            sphere_source.SetThetaResolution(5)
            sphere_source.SetPhiResolution(5)

            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow sphere

            renderer.AddActor(sphere_actor)

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(title)
    render_window.SetSize(800, 600)

    # Create an interactor and set an improved interaction style
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Use the trackball camera style for intuitive interaction
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)

    # Optional: Add an orientation axes widget
    axes = vtk.vtkAxesActor()
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(interactor)
    axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)  # Small axes in the bottom-left corner
    axes_widget.SetEnabled(1)
    axes_widget.InteractiveOn()

    camera = renderer.GetActiveCamera()
    if camera_start is not None:
        camera.SetPosition(*camera_start)
        camera.SetViewUp(0, 1, 0)
    if camera_focus is not None:
        camera.SetFocalPoint(*camera_focus)
        camera.SetViewUp(0, 1, 0)

    # Start the rendering and interaction loop
    render_window.Render()
    interactor.Initialize()  # Ensure interactor is properly initialized
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
