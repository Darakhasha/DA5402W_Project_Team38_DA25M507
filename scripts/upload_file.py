import argparse
import os
import sys
from s3_utils import upload_file ,download_file # Import your MinIO storage function

def main():
    parser = argparse.ArgumentParser(
        description="Upload ML model artifacts to MinIO S3 storage."
    )

    # Command-line arguments
    parser.add_argument(
        "-b", "--bucket",
        type=str,
        default="data-files",
        help="Target MinIO bucket name (default: TAXI_BUCKET_NAME env or 'data-files')"
    )
    parser.add_argument(
        "-f", "--filepath",
        type=str,
        default="dummy/training_data.csv",
        help="Path to the local file to upload (default: 'dummy/training_data.csv')"
    )
    parser.add_argument(
            "-d", "--dest-path",
            type=str,
            default="data/training_data.csv",
            help="Destination Path to upload (default: 'data/training_data.csv')"
        )
    # parser.add_argument(
    #     "-s", "--commit-sha",
    #     type=str,
    #     default=os.getenv("IMAGE_TAG", "latest"),
    #     help="Git Commit SHA or version tag (default: IMAGE_TAG env or 'latest')"
    # )
    # parser.add_argument(
    #     "--skip-versioned",
    #     action="store_true",
    #     help="If set, only uploads the 'latest' copy and skips the SHA-versioned copy."
    # )

    args = parser.parse_args()

    # Validate local file exists
    if not os.path.exists(args.filepath):
        print(f"Error: Local file '{args.filepath}' not found.", file=sys.stderr)
       
        bucket =  args.bucket
        object_name = args.dest_path
        local_path = args.filepath

       
        download_file(bucket=bucket, object_name=object_name, local_path=local_path)
        sys.exit(1)

    # 1. Always upload/overwrite the 'latest' file
    print(f"--> Uploading latest file to bucket '{args.bucket}'...")
    upload_file(
        local_path=args.filepath,
        bucket=args.bucket,
        object_name=args.dest_path
    )

    # # 2. Upload unique versioned copy using Git SHA (unless skipped)
    # if not args.skip_versioned:
    #     versioned_name = f"models/taxi_model_{args.commit_sha}.pkl"
    #     print(f"--> Uploading versioned file ({args.commit_sha}) to bucket '{args.bucket}'...")
    #     upload_file(
    #         local_path=args.filepath,
    #         bucket=args.bucket,
    #         object_name=versioned_name
    #     )

if __name__ == "__main__":
    main()
    