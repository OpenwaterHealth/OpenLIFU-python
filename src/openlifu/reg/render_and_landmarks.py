# render_and_landmarks.py

import vtk
import cv2
import mediapipe as mp
from vtk.util.numpy_support import vtk_to_numpy


class MeshRenderer:
    def render_mesh_at_angles(self, mesh, angles, image_size=(512, 512), flip_image=True, skin_color=(0.91, 0.76, 0.65), debug=False):
        """Render the mesh at different angles with a skin tone color and return the images."""
        images = []
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.SetSize(image_size[0], image_size[1])
        render_window.AddRenderer(renderer)
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(skin_color)
        renderer.AddActor(actor)

        for angle in angles:
            camera = renderer.GetActiveCamera()
            
            # Apply rotations around Y (Azimuth), X (Elevation), and Z (Roll) axes
            camera.Azimuth(angle[0])   # Rotate around Y-axis
            camera.Elevation(angle[1]) # Rotate around X-axis
            #camera.Roll(angle[2])      # Rotate around Z-axis (Roll)

            renderer.ResetCamera()

            render_window.Render()
            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(render_window)
            window_to_image_filter.Update()

            vtk_image = window_to_image_filter.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
            img = vtk_array.reshape((height, width, -1))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if flip_image:
                img = cv2.flip(img, 0)
            images.append(img)

            if debug:
                cv2.imshow("Rendered Image", img)
                cv2.waitKey(0)

        return images

    def animate(self, obj, event):
        """Callback function for the VTK animation."""
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(1)  # Adjust rotation speed as needed
        self.render_window.Render()

    def start_animation(self):
        """Starts the continuous rendering animation."""
        animationScene = vtk.vtkAnimationScene()
        animationScene.SetModeToRealTime()
        animationScene.SetLoop(0)  # Loop indefinitely

        animationCue = vtk.vtkAnimationCue()
        animationCue.SetStartTime(0)
        animationCue.SetEndTime(10000)  # Long end time
        animationCue.SetFrameRate(30)  # Adjust as needed
        animationCue.SetAnimationTypeToFunction()
        animationCue.SetCallback(self.animate)

        animationScene.AddCue(animationCue)
        animationScene.Start()

class LandmarkDetector:
    def detect_face_landmarks(self, images):
        """Detect facial landmarks in the rendered images and return the image with the highest confidence."""
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        annotated_images = []
        all_landmarks_2d = []
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        highest_confidence = -1  # To track the highest confidence value
        best_image_index = -1  # To track the index of the image with the highest confidence
        best_annotated_image = None  # To store the annotated image with the highest confidence
        best_landmarks_2d = []  # To store the landmarks of the image with the highest confidence

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.1) as face_mesh:
            for idx, image in enumerate(images):
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # If landmarks are found, check the confidence score and update if it's the highest
                if results.multi_face_landmarks and results.multi_face_landmarks[0]:
                    face_landmarks = results.multi_face_landmarks[0]

                    # For now, the detection confidence is part of the face_mesh process
                    # In the future, you may want to modify the model to expose this more clearly
                    detection_confidence = results.multi_face_landmarks[0].landmark[0].visibility

                    # Update the best detection based on confidence score
                    if detection_confidence > highest_confidence:
                        highest_confidence = detection_confidence
                        best_image_index = idx

                        # Copy the current image for annotation
                        best_annotated_image = image.copy()

                        # Annotate the image with the landmarks
                        mp_drawing.draw_landmarks(
                            image=best_annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=best_annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=best_annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                        # Extract the 2D landmark positions
                        best_landmarks_2d = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in face_landmarks.landmark]

                    # Append the annotated image regardless of confidence to the list
                    annotated_images.append(best_annotated_image)

                else:
                    # If no landmarks are detected, append the original image unaltered
                    annotated_images.append(image)

        # Return the image index, annotated image, and 2D landmarks with the highest confidence
        return best_image_index, best_annotated_image, best_landmarks_2d

    def unproject_2d_to_3d(self, camera, renderer, screen_points, depth_value=0.5):
        """Unproject 2D screen points to 3D points in the volume's coordinate space."""
        unprojected_points = []

        for screen_point in screen_points:
            x, y = screen_point
            
            # Normalize screen coordinates to [-1, 1] range
            norm_x = 2.0 * x / renderer.GetSize()[0] - 1.0
            norm_y = 2.0 * y / renderer.GetSize()[1] - 1.0

            # Use vtk's method to convert screen to world coordinates
            world_coords = [0.0, 0.0, 0.0]
            camera.ViewToWorld(norm_x, norm_y, depth_value, world_coords)

            unprojected_points.append(world_coords[:3])

        return unprojected_points

    def render_landmarks_in_volume(self, renderer, landmarks_3d, landmark_color=(1, 0, 0)):
        """Render the 3D landmarks inside the volume using VTK."""
        for point in landmarks_3d:
            # Create a sphere or point at each landmark position
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(point[0], point[1], point[2])
            sphere_source.SetRadius(1.0)  # Adjust radius as needed

            # Create a mapper and actor for the sphere
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere_source.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(landmark_color)

            # Add the actor to the renderer
            renderer.AddActor(actor)

        return renderer