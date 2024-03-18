import boto3
import cv2
import numpy as np
from ultralytics import YOLO


# Initialize AWS credentials and Kinesis Video client
region_name = 'us-east-1'
stream_name = 'camera_100'

session = boto3.Session()
kinesis_video_client = session.client('kinesisvideo', region_name=region_name)


def get_kinesis_video_stream_frames(client, stream_name):

    response = client.get_data_endpoint(
        StreamName=stream_name,
        APIName='GET_MEDIA'
    )

    video_data_endpoint = response['DataEndpoint']

    # Create a Kinesis video media client
    kinesis_video_media_client = session.client('kinesis-video-media',
                                                region_name=region_name,
                                                endpoint_url=video_data_endpoint)

    # Retrieve video stream
    response = kinesis_video_media_client.get_media(
        StreamName=stream_name,
        StartSelector={'StartSelectorType': 'NOW'}
    )

    return response['Payload'].read()


def load_model(model_path):
    model = YOLO(model_path)
    return model


def object_detector(frame_data, model_path):
    # Encoding the frame into numpy array
    nparr = np.frombuffer(frame_data, np.uint8)

    # Decode the frame
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Processing the Frame with YOLO model
    # You can replace this with actual YOLO inference code
    model = YOLO(model_path)
    results = model(frame)[0]
    # Draw bounding boxes and labels

    img_objects = []
    for box, conf, class_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = box
        if conf > 0.90:
            img_objects.append(model_path.names[int(class_id)].upper())
    if 'dog' in img_objects or 'person' in img_objects:
        return True
    else:
        return False


model = load_model('model/best.pt')
frames = get_kinesis_video_stream_frames(kinesis_video_client, stream_name)
object_detector(frames, model)
