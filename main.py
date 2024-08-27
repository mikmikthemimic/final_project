import os
import cv2

from tracker import Tracker

video_path = os.path.join('.', 'data', 'insert vid here.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                            (frame.shape[1], frame.shape[0]))

model = 'insert model here'

tracker = Tracker()

#insert colors
# colors =

detection_threshold = 0.5

while ret:
    #results = model(frame)
    # vv this is the same as ^^
    model.eval()
    for sample in dataloader_prediction:
        x, x_name = sample
        with torch.no_grad():
            pred = model(x)
            # Move tensors to CPU before converting to NumPy
            pred = {key: value.cpu().numpy() for key, value in pred[0].items()}
            name = pathlib.Path(x_name[0])
            save_dir = pathlib.Path(os.getcwd()) / params["PREDICTIONS_PATH"]
            save_dir.mkdir(parents=True, exist_ok=True)
            pred_list = {
                key: value.tolist() for key, value in pred.items()
            }  # numpy arrays are not serializable -> .tolist()
            save_json(pred_list, path=save_dir / name.with_suffix(".json"))

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            class_id = int(class_id)

            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

            tracker.update(frame, detections)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                image = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2), 1))
                
    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
# cv2.destroyAllWindows()