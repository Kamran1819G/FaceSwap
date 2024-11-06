import customtkinter as ctk
import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox


class FaceSwapApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Face Swap App")
        self.window.geometry("1200x800")

        # Initialize face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")

        # Image variables
        self.source_image = None
        self.target_image = None
        self.result_image = None

        self.create_gui()

    def create_gui(self):
        # Create main frame
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Buttons frame
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(pady=10)

        # Load buttons
        ctk.CTkButton(
            button_frame,
            text="Load Source Image",
            command=lambda: self.load_image("source")
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Load Target Image",
            command=lambda: self.load_image("target")
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Swap Faces",
            command=self.swap_faces
        ).pack(side="left", padx=5)

        # Images frame
        images_frame = ctk.CTkFrame(main_frame)
        images_frame.pack(pady=10, fill="both", expand=True)

        # Create labels for images
        self.source_label = ctk.CTkLabel(images_frame, text="Source Image")
        self.source_label.pack(side="left", padx=10, expand=True)

        self.target_label = ctk.CTkLabel(images_frame, text="Target Image")
        self.target_label.pack(side="left", padx=10, expand=True)

        self.result_label = ctk.CTkLabel(images_frame, text="Result")
        self.result_label.pack(side="left", padx=10, expand=True)

    def load_image(self, image_type):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image_type == "source":
                self.source_image = image
                self.update_preview(self.source_label, image)
            else:
                self.target_image = image
                self.update_preview(self.target_label, image)

    def update_preview(self, label, image):
        # Resize image for preview
        height, width = image.shape[:2]
        max_size = 300
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        # Convert to PhotoImage
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo

    def get_landmarks(self, image, face_rect):
        landmarks = self.predictor(image, face_rect)
        return np.array([[p.x, p.y] for p in landmarks.parts()])

    def swap_faces(self):
        if self.source_image is None or self.target_image is None:
            messagebox.showerror("Error", "Please load both images first")
            return

        # Convert images to grayscale for face detection
        source_gray = cv2.cvtColor(self.source_image, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2GRAY)

        # Detect faces
        source_faces = self.detector(source_gray)
        target_faces = self.detector(target_gray)

        if not source_faces or not target_faces:
            messagebox.showerror(
                "Error", "No faces detected in one or both images")
            return

        # Get landmarks for first face in each image
        source_landmarks = self.get_landmarks(source_gray, source_faces[0])
        target_landmarks = self.get_landmarks(target_gray, target_faces[0])

        # Calculate mask for seamless cloning
        hull = cv2.convexHull(target_landmarks)
        mask = np.zeros(self.target_image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # Calculate transformation matrix
        transformation = cv2.estimateAffinePartial2D(
            source_landmarks, target_landmarks)[0]

        # Warp source face
        warped_source = cv2.warpAffine(
            self.source_image,
            transformation,
            (self.target_image.shape[1], self.target_image.shape[0]),
            borderMode=cv2.BORDER_REFLECT
        )

        # Seamless clone
        center = (int(np.mean(target_landmarks[:, 0])), int(
            np.mean(target_landmarks[:, 1])))
        self.result_image = cv2.seamlessClone(
            warped_source,
            self.target_image,
            mask,
            center,
            cv2.NORMAL_CLONE
        )

        # Update result preview
        self.update_preview(self.result_label, self.result_image)

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = FaceSwapApp()
    app.run()
