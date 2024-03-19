import time
import boto3
import cv2
import numpy as np
from ultralytics import YOLO


# Initialize AWS credentials and Kinesis Video client
region_name = 'us-east-1'
stream_name = 'camera_100'
model_path= "model/best.pt"

session = boto3.Session()
kinesis_video_client = session.client('kinesisvideo', region_name=region_name)

kvs= boto3.client("kinesisvideo")
endpoint= kvs.get_data_endpoint(
    APINAME="GET_HLS_STREAMING_SESSION_URL",
    stream_name=stream_name
)["DataEndpoint"]

print(endpoint)
kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint)

url = kvam.get_hls_stream_session_url(stream_name=stream_name,PlaybackMode="LIVE")["HLSStreamingSessionURL"]

vcap = cv2.VideoCapture(url)

while (True):
    ret, frame = vcap.read()
    if frame is not None:
        model = YOLO(model_path)
        results = model(frame)[0]
        # Draw bounding boxes and labels

        img_objects = []
        for box, conf, class_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = box
            if conf > 0.90:
                img_objects.append(model_path.names[int(class_id)].upper())
        if 'dog' in img_objects or 'person' in img_objects:
            client = boto3.client("sns")
            client.publish(
                TopicArn="arn:aws:sns:us-east-1:905418005302:dog_detection:10e1c03a-8211-49d6-bc94-051c196651a0",
                Message="A dog was detected alone",
            )
        if cv2.waitKey(22)& 0xFF == ord("q"):
            break
        else:
            print("frame is None")
            break

vcap.release()
cv2.destroyAllWindows()
print("Video stream is closed")
