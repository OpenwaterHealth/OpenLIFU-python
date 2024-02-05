import vtk, slicer, mrml


segfilepath = "/Users/kedar/Desktop/Sub18-22_output/OW_018_petra.nii"
filepath = "/Users/kedar/Desktop/Sub18-22_output/OW_018_petra/OW_018_petra.nrrd"

class MeshRenderer:
    def __init__(self):
        self.points = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        self.mesh = vtk.vtkPolyData()
        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
    
    def add_point(self, x, y, z):
        self.points.InsertNextPoint(x, y, z)
    
    def add_cell(self, p1, p2, p3, p4):
        quad = vtk.vtkQuad()
        quad.GetPointIds().SetId(0, p1)
        quad.GetPointIds().SetId(1, p2)
        quad.GetPointIds().SetId(2, p3)
        quad.GetPointIds().SetId(3, p4)
        self.cells.InsertNextCell(quad)
    
    def create_mesh(self):
        self.mesh.SetPoints(self.points)
        self.mesh.SetPolys(self.cells)
    
    def render_mesh(self):
        self.mapper.SetInputData(self.mesh)
        self.actor.SetMapper(self.mapper)
        
        renderer = vtk.vtkRenderer()
        renderer.AddActor(self.actor)
        
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        render_window.Render()
        interactor.Start()

class SlicerInterface:
    def __init__(self):
        self.volume_node = None
    
    def read_and_display_volume(self, filepath):
        # Read the NRRD volume from disk
        reader = vtk.vtkNrrdReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        # Create a new volume node in 3D slicer
        self.volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        self.volume_node.SetAndObserveImageData(reader.GetOutput())
        display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeDisplayNode")
        self.volume_node.SetAndObserveDisplayNodeID(display_node.GetID())
        slicer.mrmlScene.AddNode(self.volume_node)
        slicer.util.setSliceViewerLayers(background=self.volume_node)

    def test_volume_and_render_mesh(self, filepath):
        # Load the NRRD volume file
        self.read_and_display_volume(filepath)
        
        # Create a mesh renderer
        mesh_renderer = MeshRenderer()
        
        # Add points for an 8x8 mesh at the origin
        for i in range(8):
            for j in range(8):
                mesh_renderer.add_point(i, j, 0)
        
        # Add cells for the mesh
        for i in range(7):
            for j in range(7):
                p1 = i * 8 + j
                p2 = p1 + 1
                p3 = p1 + 9
                p4 = p1 + 8
                mesh_renderer.add_cell(p1, p2, p3, p4)
        
        # Create and render the mesh
        mesh_renderer.create_mesh()
        #mesh_renderer.render_mesh()

def main():
    slicer_interface = SlicerInterface()
    slicer_interface.read_and_display_volume(segfilepath)
    #slicer_interface.test_render_mesh(filepath)

if __name__ == "__main__":
    main()
