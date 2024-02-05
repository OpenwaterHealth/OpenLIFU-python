import slicer, vtk
import numpy as np

filepath1 = "/Users/kedar/Downloads/mni_base.nii"
filePath2 = "/Users/kedar/Downloads/pnp_base.nii"
segfilepath = "/Users/kedar/Downloads/MASK_mni_base.nii"

# Display the volume as background
volumeNode = slicer.util.loadVolume(filepath1)
slicer.util.setSliceViewerLayers(background=volumeNode)

displayNode = volumeNode.GetDisplayNode()
if displayNode is None:
    displayNode = slicer.vtkMRMLScalarVolumeDisplayNode()
    slicer.mrmlScene.AddNode(displayNode)
    volumeNode.SetAndObserveDisplayNodeID(displayNode.GetID())

foregroundNode = slicer.util.loadVolume(filePath2)
slicer.util.setSliceViewerLayers(background=volumeNode, foreground=foregroundNode)

#Set alpha blending of foreground and background to 0.5 in all views
for sliceViewName in slicer.app.layoutManager().sliceViewNames():
    sliceWidget = slicer.app.layoutManager().sliceWidget(sliceViewName)
    sliceLogic = sliceWidget.sliceLogic()
    sliceCompositeNode = sliceLogic.GetSliceCompositeNode()
    sliceCompositeNode.SetForegroundOpacity(0.5)




foregroundDisplayNode = foregroundNode.GetDisplayNode()
if foregroundDisplayNode is None:
    foregroundDisplayNode = slicer.vtkMRMLScalarVolumeDisplayNode()
    slicer.mrmlScene.AddNode(foregroundDisplayNode)
    foregroundNode.SetAndObserveDisplayNodeID(foregroundDisplayNode.GetID())

#Blend the two volumes
slicer.util.setSliceViewerLayers(background=volumeNode, foreground=foregroundNode)


#foregroundDisplayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeViridis")



# Load the segmentation file
segmentationNode = slicer.util.loadSegmentation(segfilepath)

# Create a display node for the segmentation
segmentationDisplayNode = segmentationNode.GetDisplayNode()
if segmentationDisplayNode is None:
    segmentationDisplayNode = slicer.vtkMRMLSegmentationDisplayNode()
    slicer.mrmlScene.AddNode(segmentationDisplayNode)
    segmentationNode.SetAndObserveDisplayNodeID(segmentationDisplayNode.GetID())

# Display four up view
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)




