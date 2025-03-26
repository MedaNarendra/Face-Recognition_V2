import cv2
import typing
import numpy as np
import asyncio

from selfieSegmentation import MPSegmentation

class Engine:
    def __init__(
        self, 
        webcam_id: int = 0,
        custom_objects: typing.Iterable = [],
        frame_interval: int = 5,  # Process one frame every 4 seconds
    ) -> None:
        self.webcam_id = webcam_id
        self.custom_objects = custom_objects
        self.frame_interval = frame_interval

    def custom_processing(self, frame: np.ndarray) -> np.ndarray:
        for custom_object in self.custom_objects:
            frame = custom_object(frame)
        return frame

    def display_frame(self, frame: np.ndarray) -> None:
        """Display the processed frame with recognition results."""
        cv2.imshow("Recognition Result", frame)
        cv2.waitKey(2000)  # Display frame for 2 seconds
        cv2.destroyAllWindows()

    async def process_webcam(self) -> None:
        while True:
            cap = cv2.VideoCapture(self.webcam_id)  # Open camera only when needed
            if not cap.isOpened():
                print(f"Webcam with ID ({self.webcam_id}) can't be opened")
                await asyncio.sleep(self.frame_interval)
                continue
            
            success, frame = cap.read()
            cap.release()  # Release camera immediately after capturing frame
            
            if not success or frame is None:
                print("Ignoring empty camera frame.")
                await asyncio.sleep(self.frame_interval)
                continue
            
            frame = self.custom_processing(frame)
            print("Frame processed.")  # Debug message to confirm processing
            
            self.display_frame(frame)  # Show only the processed frame with recognition results
            
            await asyncio.sleep(self.frame_interval)  # Non-blocking wait for next capture

    def run(self):
        asyncio.run(self.process_webcam())
