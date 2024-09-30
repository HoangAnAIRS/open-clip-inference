import os
import boto3
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_directory, bucket, s3_directory):
    s3 = boto3.client('s3')

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_directory, relative_path)

            try:
                print(f"Uploading {local_path} to {bucket}/{s3_path}")
                s3.upload_file(local_path, bucket, s3_path)
            except FileNotFoundError:
                print(f"The file {local_path} was not found")
            except NoCredentialsError:
                print("Credentials not available")
                return False
    return True

# Usage
local_directory = "/mnt/images/output"
bucket_name = 'machine-learning-storage-dev'
s3_directory = "image-search/clip_embeddings"  # The prefix in your S3 bucket

success = upload_to_s3(local_directory, bucket_name, s3_directory)

if success:
    print("Upload completed successfully")
else:
    print("Upload failed")