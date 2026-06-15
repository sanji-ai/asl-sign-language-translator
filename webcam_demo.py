from __future__ import annotations

from pathlib import Path

import cv2

from src.predict import DEFAULT_IMAGE_SIZE, load_labels, load_model, predict_frame


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "asl_mobilenetv2.keras"
LABELS_PATH = ROOT / "models" / "labels.json"


def main() -> None:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Run the notebook first so models/asl_mobilenetv2.keras and models/labels.json exist."
        )

    model = load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        label, confidence, _ = predict_frame(model, labels, frame, image_size=DEFAULT_IMAGE_SIZE)
        text = f"{label} ({confidence:.1%})"

        cv2.rectangle(frame, (12, 12), (310, 68), (37, 99, 235), thickness=-1)
        cv2.putText(
            frame,
            text,
            (24, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("ASL Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
