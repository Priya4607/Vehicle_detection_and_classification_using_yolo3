import os

import cv2
import toml

from utils import Classifier


def __main():
    config = toml.load("config.toml")["Default"]
    input_path = os.path.expanduser(config["input_path"])

    config_path = os.path.expanduser(config["config_path"])
    assert config_path.endswith(".cfg"), f"{config_path} is not a .cfg file"
    weights_path = os.path.expanduser(config["weights_path"])
    assert weights_path.endswith(".weights"), f"{weights_path} is not a .weights file"
    labels_path = os.path.expanduser(config["labels_path"])
    classify = Classifier(
        config_path,
        weights_path,
        labels_path,
        config["class_index"],
        config.get("threshold", 0.5),
        config.get("confidence", 0.3),
    )
    if input_path.endswith((".jpg", ".png", ".jpeg")):

        image = cv2.imread(input_path)
        classified_frame = classify.post_process(image, False)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)
        cv2.imwrite("out.jpg", classified_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif input_path.endswith((".mp4", ".avi", ".mkv", ".mov")):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        writer = cv2.VideoWriter(
            "out.mp4", fourcc, 30, (frame_width, frame_height), True
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            classified_frame = classify.post_process(frame, True)

            writer.write(classified_frame)
        cap.release()
        writer.release()


if __name__ == "__main__":
    __main()
