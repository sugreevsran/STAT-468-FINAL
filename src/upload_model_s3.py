import os, boto3
from pathlib import Path

MODEL_PATH = Path("artifacts/model/mp_value_ridge_pipeline.joblib")
S3_BUCKET  = os.environ.get("S3_BUCKET")         
S3_KEY     = os.environ.get("S3_MODEL_KEY", "models/mp_value_ridge_pipeline.joblib")

def main():
    if not S3_BUCKET:
        raise SystemExit("Set S3_BUCKET env var.")
    s3 = boto3.client("s3")
    s3.upload_file(str(MODEL_PATH), S3_BUCKET, S3_KEY)
    print("Uploaded:", MODEL_PATH, "-> s3://%s/%s" % (S3_BUCKET, S3_KEY))

if __name__ == "__main__":
    main()
