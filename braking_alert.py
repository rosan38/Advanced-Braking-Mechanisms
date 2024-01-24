import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from playsound import playsound
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# Constants
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "SPEED": 15,
    "DETECTION_THRESHOLD": 0.7,
    "OBJECT_WIDTH_CM": 50,
    "FOCAL_LENGTH_PIXELS": 1000,
    "WARNING_THRESHOLD_IMMEDIATE": 500,
    "ROI_WIDTH": 600,
    "ROI_HEIGHT": 250,
    "MODEL_PATH": ROOT_PATH + '/pedestrian_detection_model.pth',
    "VIDEO_SIZE": (640, 480),
    "NUM_CLASSES": 2,
    "OUTPUT_PATH": ROOT_PATH + '/test-videos-output/',
    "SOUND": ROOT_PATH + '/pedestrian_alert.mp3'
}


class PedestrianDetectionApp:
    def __init__(self, root):
        self.processing_label = None
        self.video_label = None
        self.video_path = None
        self.params = None
        self.root = root
        self.model = None
        self.cap = None
        self.pause_video = False
        self.trackers = {}
        self.frame_count = 0
        self.skip_frames = 10
        self.setup_ui()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def setup_ui(self):
        self.root.title("Pedestrian Detection in Video")
        self.root.state('zoomed')

        # Parameter settings
        self.params = {
            "SPEED(km/h)": tk.DoubleVar(value=DEFAULTS["SPEED"]),
            "DETECTION_THRESHOLD": tk.DoubleVar(value=DEFAULTS["DETECTION_THRESHOLD"]),
            "OBJECT_WIDTH_CM": tk.DoubleVar(value=DEFAULTS["OBJECT_WIDTH_CM"]),
            "FOCAL_LENGTH_PIXELS": tk.DoubleVar(value=DEFAULTS["FOCAL_LENGTH_PIXELS"]),
            "WARNING_THRESHOLD_IMMEDIATE": tk.IntVar(value=DEFAULTS["WARNING_THRESHOLD_IMMEDIATE"]),
            "ROI_WIDTH": tk.IntVar(value=DEFAULTS["ROI_WIDTH"]),
            "ROI_HEIGHT": tk.IntVar(value=DEFAULTS["ROI_HEIGHT"])
        }

        for param, var in self.params.items():
            frame = tk.Frame(self.root)
            frame.pack(fill='x', padx=5, pady=5)
            label = tk.Label(frame, text=param)
            label.pack(side=tk.LEFT)
            entry = tk.Entry(frame, textvariable=var)
            entry.pack(side=tk.RIGHT, expand=True, fill='x')

        # Video selection
        self.video_path = tk.StringVar(self.root)
        # Video controls frame
        video_controls_frame = tk.Frame(self.root)
        video_controls_frame.pack(fill='x', padx=5, pady=5)

        # Browse Video Button
        tk.Button(video_controls_frame, text="Browse Video", command=self.browse_video).pack(side=tk.LEFT, padx=5)

        # Video Path Label
        tk.Label(video_controls_frame, textvariable=self.video_path).pack(side=tk.LEFT, expand=True, fill='x', padx=5)

        # Start Live Tracking Button
        tk.Button(video_controls_frame, text="Start Live Tracking", command=self.start_live_tracking).pack(side=tk.LEFT,
                                                                                                           padx=5)

        # Export Video Button
        tk.Button(video_controls_frame, text="Export Video", command=self.export_video).pack(side=tk.LEFT, padx=5)

        # Video display label
        self.video_label = tk.Label(self.root, text="Select a video to start processing")
        self.video_label.pack(fill='x', padx=5, pady=5)

    def start_live_tracking(self):
        self.pause_video = False
        self.process_video()

    def export_video(self):
        video_path = self.video_path.get()
        if not video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return

            # Display a processing message
        self.processing_label = tk.Label(self.root, text="Processing video, please wait... (0%)")
        self.processing_label.pack()
        self.root.update()

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(os.path.dirname(DEFAULTS["OUTPUT_PATH"]), f"{base_name}_processed.mp4")

        self.pause_video = False
        self.process_and_export_video(video_path, output_video_path)

        self.processing_label.destroy()

    def load_model(self):
        try:
            print("Loading model...")
            model = fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, DEFAULTS["NUM_CLASSES"])
            model.load_state_dict(torch.load(DEFAULTS["MODEL_PATH"]))
            model = model.to(self.device)
            print("Transferring model to device:", self.device)
            model.eval()
            self.model = model
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def browse_video(self):
        filename = filedialog.askopenfilename(title="Select Video", filetypes=[("MP4 files", "*.mp4")])
        if filename:
            self.video_path.set(filename)
            self.video_label.pack_forget()

    def process_video(self):
        video_path = self.video_path.get()
        if not video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Error opening video file.")
            return

        self.update_frame()

    def update_frame(self):
        detected = False
        if self.pause_video or not self.cap.isOpened():
            return

        self.frame_count += 1
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            messagebox.showinfo("Info", "Video processing completed.")
            self.reset_application()
            return

        # Check if the current frame should be skipped
        if self.frame_count % self.skip_frames != 0:
            # Recursive call to update_frame after a short delay for skipped frames
            self.video_label.after(10, self.update_frame)
            return

        # Resize frame for consistent processing
        frame = cv2.resize(frame, DEFAULTS["VIDEO_SIZE"])
        roi_width = self.params["ROI_WIDTH"].get()
        roi_height = self.params["ROI_HEIGHT"].get()
        start_x = (DEFAULTS["VIDEO_SIZE"][0] - roi_width) // 2
        roi_points = np.array(
            [[start_x, DEFAULTS["VIDEO_SIZE"][1]],
             [start_x, DEFAULTS["VIDEO_SIZE"][1] - roi_height],
             [start_x + roi_width, DEFAULTS["VIDEO_SIZE"][1] - roi_height],
             [start_x + roi_width, DEFAULTS["VIDEO_SIZE"][1]]], np.int32)

        # Detect objects in the frame
        predictions = self.detect_objects(frame)
        cv2.polylines(frame, [roi_points], True, (0, 200, 0), 2)

        for element in range(len(predictions['boxes'])):
            score = predictions['scores'][element]
            if score > self.params["DETECTION_THRESHOLD"].get():
                box = predictions['boxes'][element].numpy()
                centroid_x, centroid_y = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

                if element not in self.trackers:
                    self.trackers[element] = self.initialize_kalman_filter()
                    self.trackers[element].x = np.array([centroid_x, centroid_y, 0., 0.])  # Initial state

                kf = self.trackers[element]
                kf.predict()
                kf.update(np.array([centroid_x, centroid_y]))

                # Use the predicted state for further processing
                predicted_x, predicted_y = int(kf.x[0]), int(kf.x[1])

                # Draw bounding box and distance
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                distance = self.calculate_distance(box)
                cv2.putText(frame, f"Dist: {distance:.2f} cm", (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if cv2.pointPolygonTest(roi_points, (predicted_x, predicted_y), False) > 0:
                    # Dynamic alert distance based on vehicle speed
                    vehicle_speed_kmph = self.params["SPEED(km/h)"].get()
                    dynamic_alert_distance = self.calculate_alert_distance(vehicle_speed_kmph)
                    immediate_warning_threshold = self.params["WARNING_THRESHOLD_IMMEDIATE"].get()
                    # Determine if an alert should be triggered
                    if distance <= dynamic_alert_distance or distance <= immediate_warning_threshold:
                        detected = True

        cv2.imshow('Pedestrian Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.pause_video = True

        if detected:
            self.pause_video = True
            playsound(DEFAULTS["SOUND"])
            self.custom_popup("Alert", "Pedestrian detected!! Please apply Brake.")

        # Recursive call to update_frame after a short delay
        self.video_label.after(10, self.update_frame)

    @staticmethod
    def calculate_alert_distance(speed_kmph):
        # Convert speed from km/h to m/s
        speed_mps = speed_kmph / 3.6
        # 1.5 seconds is a common assumption reaction time
        reaction_time = 1.5
        reaction_distance = speed_mps * reaction_time
        # 6.5 m/s^2 average deceleration rate for a dry road
        deceleration = 6.5
        braking_distance = (speed_mps ** 2) / (2 * deceleration)
        total_stopping_distance = reaction_distance + braking_distance
        return total_stopping_distance * 100

    def detect_objects(self, frame):
        # Transform the frame to a tensor
        image = F.to_tensor(frame).to(self.device)
        image = image.unsqueeze(0)
        with torch.no_grad():
            predictions = self.model(image)
        for key in predictions[0]:
            if isinstance(predictions[0][key], torch.Tensor):
                predictions[0][key] = predictions[0][key].to('cpu')
        return predictions[0]

    def calculate_distance(self, box):
        x1, y1, x2, y2 = box
        return (self.params["OBJECT_WIDTH_CM"].get() * self.params["FOCAL_LENGTH_PIXELS"].get()) / (x2 - x1)

    def reset_application(self):
        # Pause video processing and release the video capture object
        self.pause_video = True
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # Reset the video capture object and path
        self.cap = None
        self.video_path.set("")

        # Reset the video label
        self.video_label.config(image='')
        self.video_label.config(text="Select a video to start processing")

    def custom_popup(self, title, message):
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("300x100")

        tk.Label(popup, text=message).pack(side="top", fill="x", pady=10)

        # Functionality for the "OK" button
        def on_ok():
            self.pause_video = False
            popup.destroy()
            self.update_frame()  # Continue processing the video

        tk.Button(popup, text="OK", command=on_ok).pack(pady=10)

        # Ensure the popup is brought to the front
        popup.lift()
        popup.attributes('-topmost', True)

    @staticmethod
    def initialize_kalman_filter():
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                         [0, 1, 0, 0]])
        kf.R *= 10  # Measurement uncertainty
        kf.P *= 1000  # Initial estimation error
        kf.Q *= 0.1  # Process uncertainty
        return kf

    def process_and_export_video(self, video_path, output_video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Error opening video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, DEFAULTS["VIDEO_SIZE"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, DEFAULTS["VIDEO_SIZE"])
            # Draw ROI and process detections
            roi_width = self.params["ROI_WIDTH"].get()
            roi_height = self.params["ROI_HEIGHT"].get()
            start_x = (DEFAULTS["VIDEO_SIZE"][0] - roi_width) // 2
            roi_points = np.array(
                [[start_x, DEFAULTS["VIDEO_SIZE"][1]],
                 [start_x, DEFAULTS["VIDEO_SIZE"][1] - roi_height],
                 [start_x + roi_width, DEFAULTS["VIDEO_SIZE"][1] - roi_height],
                 [start_x + roi_width, DEFAULTS["VIDEO_SIZE"][1]]], np.int32)

            predictions = self.detect_objects(frame)
            cv2.polylines(frame, [roi_points], True, (0, 200, 0), 2)

            # Process each detection
            for element in range(len(predictions['boxes'])):
                score = predictions['scores'][element]
                if score > self.params["DETECTION_THRESHOLD"].get():
                    box = predictions['boxes'][element].numpy()
                    centroid_x, centroid_y = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

                    # Initialize or update tracker
                    if element not in self.trackers:
                        self.trackers[element] = self.initialize_kalman_filter()
                        self.trackers[element].x = np.array([centroid_x, centroid_y, 0., 0.])  # Initial state

                    kf = self.trackers[element]
                    kf.predict()
                    kf.update(np.array([centroid_x, centroid_y]))

                    # Use the predicted state for further processing
                    predicted_x, predicted_y = int(kf.x[0]), int(kf.x[1])

                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    distance = self.calculate_distance(box)
                    cv2.putText(frame, f"Dist: {distance:.2f} cm", (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if cv2.pointPolygonTest(roi_points, (predicted_x, predicted_y), False) > 0:
                        vehicle_speed_kmph = self.params["SPEED(km/h)"].get()
                        dynamic_alert_distance = self.calculate_alert_distance(vehicle_speed_kmph)
                        immediate_warning_threshold = self.params["WARNING_THRESHOLD_IMMEDIATE"].get()
                        # Determine if an alert should be triggered
                        if distance <= dynamic_alert_distance or distance <= immediate_warning_threshold:
                            cv2.putText(frame, 'Pedestrian detected!! Please apply Brake.', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            processed_frames += 1
            percent_complete = (processed_frames / total_frames) * 100
            self.processing_label.config(text=f"Processing video, please wait... ({percent_complete:.2f}%)")
            self.root.update()
            out.write(frame)

        cap.release()
        out.release()
        messagebox.showinfo("Info", "Video export completed.")


if __name__ == "__main__":
    root = tk.Tk()
    app = PedestrianDetectionApp(root)
    root.mainloop()
