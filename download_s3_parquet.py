import argparse
import pandas as pd
import os
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

class S3Downloader:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        self.bucket_name = bucket_name

    def generate_s3_key(self, item_id, suffix):
        item_id_str = str(item_id).zfill(9)
        return f"{item_id_str[:3]}/{item_id_str[3:6]}/{item_id_str[6:]}/{item_id}_{suffix}.jpg"

    def download_file_from_s3(self, item_id, download_folder, log):
        for suffix in ['0', '1']:
            s3_key = self.generate_s3_key(item_id, suffix)
            local_filename = os.path.join(download_folder, s3_key)
            local_dir = os.path.dirname(local_filename)
            try:
                if os.path.exists(local_filename):
                    print(f'{{"{item_id}":"{s3_key}"}}', file=log)
                    break
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                self.s3_client.download_file(self.bucket_name, s3_key, local_filename)
                print(f'{{"{item_id}":"{s3_key}"}}', file=log)
                break  # Break if download succeeds
            except self.s3_client.exceptions.NoSuchKey:
                print(f"{s3_key} not found, trying next suffix.", file=log)
                continue  # Try next suffix if file not found
            except Exception as e:
                print(e, file=log)

def process_parquet_in_chunks(parquet_file_path, chunk_size):
    parquet_file = pd.read_parquet(parquet_file_path, engine='pyarrow')
    total_rows = len(parquet_file)
    for start in range(0, total_rows, chunk_size):
        end = start + chunk_size
        # Convert to list of lists for compatibility
        yield parquet_file.iloc[start:end].values.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from S3 based on item IDs in a Parquet file")
    
    parser.add_argument("--parquet_file_path", required=True, help="Path to the Parquet file with item IDs")
    parser.add_argument("--download_folder", required=True, help="Folder to download the files to")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of download threads")
    
    args = parser.parse_args()

    downloader = S3Downloader(
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        bucket_name="pixta-image-product-jp"
    )

    log_file_path = f"{args.parquet_file_path}_log.txt"
    with open(log_file_path, 'a') as log_file:
        for chunk in process_parquet_in_chunks(args.parquet_file_path, 100000):
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [
                    executor.submit(
                        downloader.download_file_from_s3,
                        row[0],  # Assuming item_id is the first column
                        args.download_folder,
                        log_file
                    )
                    for row in chunk
                ]

                for future in as_completed(futures):
                    # Optionally handle results or exceptions
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error: {e}", file=log_file)

            gc.collect()  # Trigger garbage collection
