import argparse
import csv
import boto3
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class S3Downloader:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name):
        self.s3_client = boto3.client('s3',
                                      aws_access_key_id=aws_access_key_id,
                                      aws_secret_access_key=aws_secret_access_key)
        self.bucket_name = bucket_name

    def generate_s3_key(self, item_id, suffix):
        item_id_str = str(item_id).zfill(9)
        return f"{item_id_str[:3]}/{item_id_str[3:6]}/{item_id_str[6:]}/{item_id}_{suffix}.jpg"

    def generate_s3_local_path(self, item_id, suffix=0):
        item_id_str = str(item_id).zfill(9)
        return f"{item_id_str[:3]}/{item_id_str[3:6]}/{item_id_str[6:]}"

    def download_file_from_s3(self, item_id, download_folder, success_log, error_log):
        for suffix in ['0','1']:
            s3_key = self.generate_s3_key(item_id, suffix)
            local_filename = os.path.join(download_folder, s3_key)
            local_dir = os.path.dirname(local_filename)
            try:
                if os.path.exists(local_filename):
                    print(f"{{\"{item_id}\":\"{s3_key}\"}}", file=success_log)
                    return True
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                self.s3_client.download_file(self.bucket_name, s3_key, local_filename)
                print(f"{{\"{item_id}\":\"{s3_key}\"}}", file=success_log)
                return True
            except self.s3_client.exceptions.NoSuchKey:
                print(f"{s3_key} not found, trying next suffix.", file=error_log)
                continue
            except Exception as e:
                print(f"Error downloading {s3_key}: {str(e)}", file=error_log)
        
        print(f"{{\"{item_id}\":\"failed\"}}", file=error_log)
        return False

def process_csv_in_chunks(csv_file_path, chunk_size):
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def count_csv_rows(csv_file_path):
    with open(csv_file_path, 'r', newline='') as file:
        return sum(1 for row in file) - 1  # Subtract 1 to account for header

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from S3 based on item IDs in a CSV file")
    
    parser.add_argument("--csv_file_path", required=True, help="Path to the CSV file with item IDs")
    parser.add_argument("--download_folder", required=True, help="Folder to download the files to")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of download threads")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of rows to process in each chunk")

    args = parser.parse_args()
    
    success_log = open(f"{args.csv_file_path}_success.log", 'a')
    error_log = open(f"{args.csv_file_path}_error.log", 'a')

    downloader = S3Downloader(os.environ.get('AWS_ACCESS_KEY'),
                              os.environ.get('AWS_SECRET_ACCESS_KEY'),
                              "pixta-image-product-jp")

    total_rows = count_csv_rows(args.csv_file_path)
    logging.info(f"Total rows to process: {total_rows}")

    processed_rows = 0
    with tqdm(total=total_rows, desc="Processing", unit="row") as pbar:
        for chunk in process_csv_in_chunks(args.csv_file_path, args.chunk_size):
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(downloader.download_file_from_s3, row[0], args.download_folder, success_log, error_log) for row in chunk]

                for future in as_completed(futures):
                    processed_rows += 1
                    pbar.update(1)

            gc.collect()  # Trigger garbage collection

            logging.info(f"Processed {processed_rows}/{total_rows} rows")
    
    success_log.close()
    error_log.close()

    logging.info(f"Download complete. Check {args.csv_file_path}_success.log for successful downloads and {args.csv_file_path}_error.log for errors.")