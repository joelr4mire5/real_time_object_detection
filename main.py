import time
import boto3
from PIL import Image
import os
from ultralytics import YOLO
import cv2

# Initialize AWS credentials and Kinesis Video client
region_name = 'us-east-1'
stream_name = 'camera_100'
model_path = "model/best.pt"

# Create a S3 resource object
s3 = boto3.resource('s3')

# Specify your S3 bucket name
bucket_name = 'cctvimagescamera100'
folder_name = 'camera_100'

# Create a Bucket resource
bucket = s3.Bucket(bucket_name)

while True:
    objects = [(obj.last_modified, obj.key) for obj in bucket.objects.filter(Prefix=folder_name)]
    objects.sort(reverse=True)
    newest_object_name = objects[0][1]
    newest_image_path = 'newest_image.jpeg'

    # Download the newest object
    bucket.download_file(newest_object_name, newest_image_path)
    time.sleep(5)  # wait for 5 seconds before attempting to open
    print('Downloaded newest_image.jpeg')

    image_path = os.path.join(os.getcwd(), newest_image_path)
    frame = cv2.imread(image_path)


    # Run image through YOLO model
    model = YOLO(model_path)
    try:
        results = model.predict(frame)[0]
    except Exception as e:
        print(f"Failed to generate prediction with error: {str(e)}")
        continue

    # Draw bounding boxes and labels
    img_objects = []
    for box, conf, class_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = box
        if conf > 0.50:
            img_objects.append(model.names[int(class_id)].upper())

    # If dog is detected, send SNS notification
    if 'DOG' in img_objects or 'PERSON' in img_objects:
        client = boto3.client("sns",
                              region_name=region_name)  # Replace 'your-region-name' with your actual region
        topic_arn = "arn:aws:sns:us-east-1:905418005302:dog_detection"
        client.publish(
            TopicArn=topic_arn,
            Message="A dog was detected alone"
        )
        print("topic published")

    # Add delay before the next iteration
    print(img_objects)
    time.sleep(2)
