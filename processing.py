import time
import boto3
import av
import numpy as np
from ultralytics import YOLO
import cv2


# Initialize AWS credentials and Kinesis Video client
region_name = 'us-east-1'
stream_name = 'camera_100'
model_path= "model/best.pt"

session = boto3.Session()
kinesis_video_client = session.client('kinesisvideo', region_name=region_name)

kvs= boto3.client("kinesisvideo", region_name=region_name)

endpoint= kvs.get_data_endpoint(
    StreamName=stream_name,
    APIName="GET_HLS_STREAMING_SESSION_URL",

)["DataEndpoint"]

print(endpoint)
kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint, region_name=region_name)

url = kvam.get_hls_streaming_session_url(StreamName=stream_name, PlaybackMode="LIVE")["HLSStreamingSessionURL"]
model = YOLO(model_path)
container = av.open(url)

stream = next(s for s in container.streams if s.type == 'video')

for frame in container.decode(stream):
    img_array = frame.to_ndarray(format='bgr24')
    results = model(img_array)[0]

    img_objects = []
    for box, conf, class_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = box
        if conf > 0.90:
            img_objects.append(model_path.names[int(class_id)].upper())
    if 'dog' in img_objects or 'person' in img_objects:
        client = boto3.client("sns", region_name=region_name)
        client.publish(
            TopicArn="arn:aws:sns:us-east-1:905418005302:dog_detection:10e1c03a-8211-49d6-bc94-051c196651a0",
            Message="A dog was detected alone",
        )
        print("message sent")
    if np.all(np.array(cv2.waitKey(22)) & 0xFF == ord("q")):
        break
    else:
        print("frame is None")
        break

print("Video stream is closed")

