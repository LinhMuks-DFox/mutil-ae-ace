#!/usr/bin/env python3
import os
import sys
import platform
import requests
import zipfile
import tarfile
import shutil
import argparse
from io import BytesIO

# --- Configuration ---
ORT_VERSION = "1.17.3"  # Specify the ONNX Runtime version you want
DEST_DIR = os.path.abspath("./vendor") # Download destination base directory
EXTRACT_DIR_TEMP = os.path.join(DEST_DIR, "onnxruntime_temp") # Temp extraction folder
FINAL_ORT_DIR = os.path.join(DEST_DIR, "onnxruntime") # Final target folder for ORT contents

# --- URL Mapping (CPU Builds) ---
# Add more platforms/architectures if needed
# Keys are tuples: (system, machine_normalized)
URL_MAP = {
    ("Windows", "x64"): f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-win-x64-{ORT_VERSION}.zip",
    ("Linux",   "x64"): f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-linux-x64-{ORT_VERSION}.tgz",
    ("Darwin",  "x64"): f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-osx-x64-{ORT_VERSION}.tgz", # macOS Intel
    ("Darwin", "arm64"): f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-osx-arm64-{ORT_VERSION}.tgz", # macOS Apple Silicon
}
# -----------------------

def normalize_machine(machine):
    """Normalize machine architecture strings."""
    machine = machine.lower()
    if machine == "amd64":
        return "x64"
    if machine == "x86_64":
        return "x64"
    if machine == "aarch64":
        return "arm64"
    # Add other normalizations if necessary
    return machine

def download_file(url, filename):
    """Downloads a file from a URL, showing progress."""
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=600) # Timeout 10 minutes
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        downloaded_size = 0

        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
                progress = int(50 * downloaded_size / total_size) if total_size else 0
                sys.stdout.write(f"\r[{'=' * progress}{' ' * (50 - progress)}] {downloaded_size / (1024*1024):.2f} / {total_size / (1024*1024):.2f} MB")
                sys.stdout.flush()
        print("\nDownload complete.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading file: {e}")
        return False
    except IOError as e:
        print(f"\nError writing file: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """Extracts zip or tgz archives."""
    print(f"Extracting {archive_path} to {extract_to}...")
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(".tgz") or archive_path.endswith(".tar.gz"):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Error: Unsupported archive format: {archive_path}")
            return False
        print("Extraction complete.")
        return True
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as e:
        print(f"Error extracting archive: {e}")
        return False
    except Exception as e:
         print(f"An unexpected error occurred during extraction: {e}")
         return False

def find_extracted_content_dir(base_extract_path):
    """Finds the actual content directory within the extracted files."""
    # Archives often contain a single top-level directory like 'onnxruntime-win-x64-1.17.3'
    extracted_items = os.listdir(base_extract_path)
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(base_extract_path, extracted_items[0])):
        # Found a single directory, assume this is the root of ORT contents
        return os.path.join(base_extract_path, extracted_items[0])
    elif all(item in extracted_items for item in ['include', 'lib']):
        # Found 'include' and 'lib' directly, use the base path
        return base_extract_path
    else:
        print(f"Warning: Could not reliably determine the ORT content directory in {base_extract_path}.")
        print(" Found items:", extracted_items)
        # Fallback: Try finding a directory starting with 'onnxruntime'
        for item in extracted_items:
             if item.startswith("onnxruntime") and os.path.isdir(os.path.join(base_extract_path, item)):
                 print(f" Assuming content directory is: {item}")
                 return os.path.join(base_extract_path, item)
        return None # Indicate failure


def main():
    ORT_VERSION = "1.17.3"
    parser = argparse.ArgumentParser(description="Download and setup ONNX Runtime C++ library.")
    parser.add_argument("--force", action="store_true", help="Force download even if destination directory exists.")
    parser.add_argument("--version", default=ORT_VERSION, help=f"Specify ONNX Runtime version (default: {ORT_VERSION})")
    args = parser.parse_args()

    # global ORT_VERSION # Allow modification via argument
    ORT_VERSION = args.version

    print(f"Targeting ONNX Runtime version: {ORT_VERSION}")

    # --- Check if already downloaded ---
    if os.path.exists(FINAL_ORT_DIR) and not args.force:
        print(f"Destination directory '{FINAL_ORT_DIR}' already exists.")
        print("Skipping download. Use --force to overwrite.")
        sys.exit(0)

    # --- Detect Platform ---
    system = platform.system()
    machine = normalize_machine(platform.machine())
    print(f"Detected Platform: {system} {machine}")

    # --- Get Download URL ---
    platform_key = (system, machine)
    if platform_key not in URL_MAP:
        print(f"Error: Unsupported platform combination: {system} {machine}")
        print("Please check URL_MAP in the script or download manually.")
        sys.exit(1)
    url = URL_MAP[platform_key].replace("{ORT_VERSION}", ORT_VERSION) # Replace version placeholder if used in URL_MAP keys later

    # --- Prepare Directories ---
    os.makedirs(DEST_DIR, exist_ok=True)
    if os.path.exists(EXTRACT_DIR_TEMP):
        print(f"Cleaning up temporary directory: {EXTRACT_DIR_TEMP}")
        shutil.rmtree(EXTRACT_DIR_TEMP)
    os.makedirs(EXTRACT_DIR_TEMP)
    if os.path.exists(FINAL_ORT_DIR):
         if args.force:
             print(f"Removing existing destination directory: {FINAL_ORT_DIR}")
             shutil.rmtree(FINAL_ORT_DIR)
         else:
             # Should have exited earlier, but double-check
             print(f"Error: Final destination '{FINAL_ORT_DIR}' exists and --force not used.")
             sys.exit(1)


    # --- Download ---
    archive_name = os.path.basename(url)
    download_path = os.path.join(DEST_DIR, archive_name)
    if not download_file(url, download_path):
        sys.exit(1)

    # --- Extract ---
    if not extract_archive(download_path, EXTRACT_DIR_TEMP):
        sys.exit(1)

    # --- Organize Files ---
    # Find the actual directory containing include/lib inside the temp extraction folder
    content_dir = find_extracted_content_dir(EXTRACT_DIR_TEMP)

    if content_dir and os.path.exists(content_dir):
         print(f"Moving content from {content_dir} to {FINAL_ORT_DIR}")
         try:
             # Move the detected content directory to the final location
             shutil.move(content_dir, FINAL_ORT_DIR)
             print(f"Successfully setup ONNX Runtime in {FINAL_ORT_DIR}")
         except Exception as e:
             print(f"Error moving extracted files: {e}")
             print(f"Please check the contents of {EXTRACT_DIR_TEMP} and move manually if needed.")
             sys.exit(1)
    else:
         print(f"Error: Could not find expected content structure in {EXTRACT_DIR_TEMP}.")
         print(f"Please check the contents and setup manually.")
         sys.exit(1)


    # --- Cleanup ---
    print(f"Cleaning up downloaded archive: {download_path}")
    os.remove(download_path)
    print(f"Cleaning up temporary directory: {EXTRACT_DIR_TEMP}")
    if os.path.exists(EXTRACT_DIR_TEMP): # Should be empty if move succeeded
         try:
             os.rmdir(EXTRACT_DIR_TEMP) # Remove empty temp dir
         except OSError:
              # If not empty (e.g., move failed partially or extra files existed), remove recursively
              print(f"Warning: Temp directory {EXTRACT_DIR_TEMP} not empty after move, removing recursively.")
              shutil.rmtree(EXTRACT_DIR_TEMP)


    print("\nONNX Runtime dependency setup complete.")

if __name__ == "__main__":
    main()