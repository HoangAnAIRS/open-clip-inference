import argparse
import pandas as pd
import os
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import glob
from tqdm import tqdm

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

    def download_file_from_s3(self, item_id, download_folder, success_log, error_log):
        for suffix in ['0', '1']:
            s3_key = self.generate_s3_key(item_id, suffix)
            local_filename = os.path.join(download_folder, s3_key)
            local_dir = os.path.dirname(local_filename)
            try:
                if os.path.exists(local_filename):
                    print(f'{{"{item_id}":"{s3_key}"}}', file=success_log)
                    return True
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                self.s3_client.download_file(self.bucket_name, s3_key, local_filename)
                print(f'{{"{item_id}":"{s3_key}"}}', file=success_log)
                return True
            except self.s3_client.exceptions.NoSuchKey:
                print(f"{s3_key} not found, trying next suffix.", file=error_log)
                continue
            except Exception as e:
                print(f"Error downloading {s3_key}: {e}", file=error_log)
        return False

def process_parquet_in_chunks(parquet_file_path, chunk_size):
    parquet_file = pd.read_parquet(parquet_file_path, engine='pyarrow')
    total_rows = len(parquet_file)
    for start in range(0, total_rows, chunk_size):
        end = start + chunk_size
        yield parquet_file.iloc[start:end].values.tolist(), total_rows
        
def process_parquet_files_in_directory(directory, downloader, download_folder, max_workers, log_folder):
    parquet_files = sorted(glob.glob(os.path.join(directory, "*.parquet")))

    with tqdm(total=len(parquet_files), desc="Processing Parquet Files", position=0) as file_pbar:
        for parquet_file_path in parquet_files:
            parquet_file_name = os.path.basename(parquet_file_path)
            success_log_path = os.path.join(log_folder, f"{parquet_file_name}_success.log")
            error_log_path = os.path.join(log_folder, f"{parquet_file_name}_error.log")
            
            with open(success_log_path, 'a') as success_log, open(error_log_path, 'a') as error_log:
                print(f"Processing file: {parquet_file_path}", file=success_log)
                
                parquet_file = pd.read_parquet(parquet_file_path, engine='pyarrow')
                total_rows = len(parquet_file)
                chunk_size = 100000

                with tqdm(total=total_rows, desc=f"Processing {parquet_file_name}", position=1, leave=False) as row_pbar:
                    for chunk, _ in process_parquet_in_chunks(parquet_file_path, chunk_size):
                        with tqdm(total=len(chunk), desc="Downloading", position=2, leave=False) as download_pbar:
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = [
                                    executor.submit(
                                        downloader.download_file_from_s3,
                                        row[0],
                                        download_folder,
                                        success_log,
                                        error_log
                                    )
                                    for row in chunk
                                ]

                                for future in as_completed(futures):
                                    try:
                                        result = future.result()
                                        download_pbar.update(1)
                                    except Exception as e:
                                        print(f"Error: {e}", file=error_log)
                                        download_pbar.update(1)

                        row_pbar.update(len(chunk))
                        gc.collect()  # Trigger garbage collection

            file_pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from S3 based on item IDs in Parquet files")

    parser.add_argument("--parquet_dir_path", required=True, help="Path to the directory containing Parquet files")
    parser.add_argument("--download_folder", required=True, help="Folder to download the files to")
    parser.add_argument("--log_folder", required=True, help="Folder to store success and error logs")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of download threads")

    args = parser.parse_args()

    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)

    downloader = S3Downloader(
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        bucket_name="pixta-image-product-jp"
    )

    process_parquet_files_in_directory(
        args.parquet_dir_path,
        downloader,
        args.download_folder,
        args.max_workers,
        args.log_folder
    )