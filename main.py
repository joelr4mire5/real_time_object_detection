
import time
import boto3
import cv2
import numpy as np
from ultralytics import YOLO
import av
from PIL import Image
# to not exceed the S3 API rate limits
import time


# Initialize AWS credentials and Kinesis Video client
region_name = 'us-east-1'
stream_name = 'camera_100'
model_path= "model/best.pt"

import boto3

# Create a S3 resource object
s3 = boto3.resource('s3')

# Specify your S3 bucket name
bucket_name = 'cctvimagescamera100'
folder_name = 'camera_100'

# Create a Bucket resource
bucket = s3.Bucket(bucket_name)

objects = [(obj.last_modified, obj.key) for obj in bucket.objects.filter(Prefix=folder_name)]

objects.sort(reverse=True)

newest_object_name = objects[0][1]
newest_image_path = 'newest_image.jpg'

# Download the newest object
bucket.download_file(newest_object_name, newest_image_path)
img = Image.open(newest_image_path)

model_path = 'model/best.pt'
model = YOLO(model_path)
results = model.predict(img)[0]

# Draw bounding boxes and labels
img_objects = []
for box, conf, class_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
    x1, y1, x2, y2 = box
    if conf > 0.90:
        img_objects.append(model.names[int(class_id)].upper())

# If dog is detected, send SNS notification
if 'dog' in img_objects or 'person' in img_objects:
    client = boto3.client("sns",
                          region_name='your-region-name')  # Replace 'your-region-name' with your actual region
    topic_arn = "arn:aws:sns:us-east-1:905418005302:dog_detection:10e1c03a-8211-49d6-bc94-051c196651a0"
    client.publish(
        TopicArn=topic_arn,
        Message="A dog was detected alone"
    )
    time.sleep(600)

