import boto3
from pathlib import Path

BUCKET_NAME = "stat468-final-project"  
DATA_DIR = Path("data/processed")

s3 = boto3.client("s3")
region = boto3.session.Session().region_name

# Create bucket 
if region == "us-east-1":
    s3.create_bucket(Bucket=BUCKET_NAME)
else:
    s3.create_bucket(
        Bucket=BUCKET_NAME,
        CreateBucketConfiguration={'LocationConstraint': region}
    )

# upload
for file in DATA_DIR.glob("*.csv"):
    s3.upload_file(str(file), BUCKET_NAME, file.name)
    print(f"Uploaded {file.name} to {BUCKET_NAME}")
