import cv2
import numpy as np


class Classifier:
    def __init__(
        self,
        config_path,
        weights_path,
        labels_path,
        class_index,
        threshold,
        confidence,
    ):
        self.class_index = class_index
        self.threshold = threshold
        self.confidence = confidence
        self.labels = open(labels_path).read().strip().split("\n")
        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255, size=(len(self.labels), 3), dtype="uint8"
        )

        self.dnn = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.layer_names = self.dnn.getLayerNames()
        self.output_layers = [
            self.layer_names[i - 1] for i in self.dnn.getUnconnectedOutLayers()
        ]

    def post_process(self, frame, write):
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.dnn.setInput(blob)
        outputs = self.dnn.forward(self.output_layers)

        bbox = []
        class_ids = []
        confidences = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id in self.class_index:
                    if confidence > self.confidence:
                        coordinates = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = coordinates.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        bbox.append([x, y, int(width), int(height)])
                        class_ids.append(class_id)
                        confidences.append(confidence)

        self.__draw(bbox, confidences, class_ids, frame)
        return frame

    def __draw(self, bbox, confidences, class_ids, frame):
        ids = cv2.dnn.NMSBoxes(bbox, confidences, self.confidence, self.threshold)
        if len(ids) > 0:
            for i in ids.flatten():
                (x, y) = (bbox[i][0], bbox[i][1])
                (w, h) = (bbox[i][2], bbox[i][3])
                color = [int(c) for c in self.colors[class_ids[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[class_ids[i]], confidences[i])
                cv2.putText(
                    frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

