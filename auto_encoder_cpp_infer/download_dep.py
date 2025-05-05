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
ORT_VERSION = "1.17.3"  # !!! 确认这个版本有你需要的 ARM 预编译包 !!!
# 你可以去 GitHub Releases 页面检查: https://github.com/microsoft/onnxruntime/releases
# 如果没有 v1.17.3 的 ARM 包，请修改为有 ARM 包的最新稳定版本号

DEST_DIR = os.path.abspath("./vendor") # 下载目标基础目录
EXTRACT_DIR_TEMP = os.path.join(DEST_DIR, "onnxruntime_temp") # 临时解压目录
FINAL_ORT_DIR = os.path.join(DEST_DIR, "onnxruntime") # 最终 ORT 文件存放目录

# --- URL Mapping (CPU Builds) - 已更新 ARM ---
# Key: (System, Normalized Machine)
URL_MAP = {
    # Desktop / Cloud
    ("Windows", "x64"):   f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-win-x64-{ORT_VERSION}.zip",
    ("Linux",   "x64"):   f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-linux-x64-{ORT_VERSION}.tgz",
    ("Darwin",  "x64"):   f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-osx-x64-{ORT_VERSION}.tgz",
    ("Darwin", "arm64"):  f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-osx-arm64-{ORT_VERSION}.tgz",
    # Raspberry Pi / Linux ARM
    ("Linux", "aarch64"): f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-linux-aarch64-{ORT_VERSION}.tgz", # 64-bit ARM
    ("Linux", "armhf"):   f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-linux-armhf-{ORT_VERSION}.tgz",   # 32-bit ARM (armv7l usually uses armhf build)
}
# -----------------------------------------

def normalize_machine(machine):
    """规范化架构名称"""
    machine = machine.lower()
    if machine == "amd64" or machine == "x86_64":
        return "x64"
    if machine == "aarch64": # Linux 64-bit ARM
        return "aarch64"
    if machine == "arm64":   # macOS 64-bit ARM uses 'arm64' in URL often
         return "arm64"
    if machine == "armv7l":  # Linux 32-bit ARM
        # ORT 通常为 armv7l hard float ABI 提供 'armhf' 构建
        return "armhf"
    # 可以根据需要添加其他架构的规范化
    return machine

def download_file(url, filename):
    """下载文件并显示简易进度条"""
    print(f"Downloading {filename} from {url}...")
    try:
        # 添加 User-Agent 可能有助于避免某些服务器阻止脚本下载
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, timeout=600, headers=headers)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        downloaded_size = 0

        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
                progress = int(50 * downloaded_size / total_size) if total_size > 0 else 0
                sys.stdout.write(f"\r[{'=' * progress}{' ' * (50 - progress)}] {downloaded_size / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB")
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
    """解压 .zip 或 .tgz 文件"""
    print(f"Extracting {archive_path} to {extract_to}...")
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(".tgz") or archive_path.endswith(".tar.gz"):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                # Ensure members are safe (avoid path traversal)
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                for member in tar_ref.getmembers():
                    member_path = os.path.join(extract_to, member.name)
                    if not is_within_directory(extract_to, member_path):
                        raise Exception(f"Attempted Path Traversal in Tar File: {member.name}")
                # If all members are safe, extract
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
    """查找解压后包含 include/lib 的实际内容目录"""
    extracted_items = os.listdir(base_extract_path)
    if not extracted_items:
        return None
    # 情况1: 解压后只有一个顶层目录 (最常见)
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(base_extract_path, extracted_items[0])):
        content_dir = os.path.join(base_extract_path, extracted_items[0])
        # 确认这个目录下有 include 和 lib
        if os.path.isdir(os.path.join(content_dir, 'include')) and os.path.isdir(os.path.join(content_dir, 'lib')):
             return content_dir
    # 情况2: include 和 lib 直接在解压根目录
    elif all(item in extracted_items for item in ['include', 'lib']):
        return base_extract_path
    # 尝试在子目录中查找 (可能有多层嵌套，这里只看一层)
    for item in extracted_items:
        potential_dir = os.path.join(base_extract_path, item)
        if os.path.isdir(potential_dir):
             if os.path.isdir(os.path.join(potential_dir, 'include')) and os.path.isdir(os.path.join(potential_dir, 'lib')):
                 return potential_dir
    # 找不到符合条件的目录
    print(f"Warning: Could not reliably determine the ORT content directory in {base_extract_path}.")
    print(" Found items:", extracted_items)
    return None


def main():
    ORT_VERSION = "1.17.3"
    parser = argparse.ArgumentParser(description="Download and setup ONNX Runtime C++ library.")
    parser.add_argument("--force", action="store_true", help="Force download even if destination directory exists.")
    # parser.add_argument("--version", default=ORT_VERSION, help=f"Specify ONNX Runtime version (default: {ORT_VERSION})")
    args = parser.parse_args()

    # ORT_VERSION = args.version

    print(f"Targeting ONNX Runtime version: {ORT_VERSION}")

    # --- 检查目标目录 ---
    if os.path.exists(FINAL_ORT_DIR) and not args.force:
        print(f"Destination directory '{FINAL_ORT_DIR}' already exists.")
        # 可以添加检查版本号的逻辑
        # with open(os.path.join(FINAL_ORT_DIR, "VERSION.txt"), 'r') as f: ...
        print("Skipping download. Use --force to overwrite.")
        sys.exit(0)

    # --- 检测平台 ---
    system = platform.system()
    machine_normalized = normalize_machine(platform.machine())
    print(f"Detected Platform: System='{system}', Machine='{platform.machine()}', Normalized='{machine_normalized}'")

    # --- 获取下载 URL ---
    platform_key = (system, machine_normalized)
    if platform_key not in URL_MAP:
        print(f"Error: Unsupported platform combination: System='{system}', Machine='{machine_normalized}'")
        print("Available platform keys in script:", list(URL_MAP.keys()))
        print(f"Please check if version {ORT_VERSION} provides a pre-built package for this platform,")
        print("or update the URL_MAP in the script, or consider building from source.")
        sys.exit(1)
    url = URL_MAP[platform_key]
    print(f"Found URL for platform {platform_key}: {url}")

    # --- 准备目录 ---
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
             print(f"Error: Final destination '{FINAL_ORT_DIR}' exists and --force not used.")
             sys.exit(1)

    # --- 下载 ---
    archive_name = os.path.basename(url)
    download_path = os.path.join(DEST_DIR, archive_name)
    if not download_file(url, download_path):
        # 清理临时目录
        shutil.rmtree(EXTRACT_DIR_TEMP)
        sys.exit(1)

    # --- 解压 ---
    if not extract_archive(download_path, EXTRACT_DIR_TEMP):
        # 清理下载的文件和临时目录
        os.remove(download_path)
        shutil.rmtree(EXTRACT_DIR_TEMP)
        sys.exit(1)

    # --- 整理文件 ---
    content_dir = find_extracted_content_dir(EXTRACT_DIR_TEMP)

    if content_dir and os.path.exists(content_dir):
         print(f"Moving content from '{content_dir}' to '{FINAL_ORT_DIR}'")
         try:
             # 将实际内容目录移动到最终目标位置
             shutil.move(content_dir, FINAL_ORT_DIR)
             # (可选) 可以在 FINAL_ORT_DIR 中写入一个版本文件
             # with open(os.path.join(FINAL_ORT_DIR, "VERSION.txt"), 'w') as f: f.write(ORT_VERSION)
             print(f"Successfully set up ONNX Runtime in '{FINAL_ORT_DIR}'")
         except Exception as e:
             print(f"Error moving extracted files: {e}")
             print(f"Please check the contents of '{EXTRACT_DIR_TEMP}' and move manually if needed.")
             # 保留临时目录以便手动检查
             sys.exit(1)
    else:
         print(f"Error: Could not find expected content structure (include/lib folders) in '{EXTRACT_DIR_TEMP}'.")
         print("Extraction might have failed or the archive structure is unexpected.")
         # 保留临时目录以便手动检查
         sys.exit(1)

    # --- 清理 ---
    print(f"Cleaning up downloaded archive: {download_path}")
    try:
        os.remove(download_path)
    except OSError as e:
        print(f"Warning: Could not remove archive file {download_path}: {e}")

    print(f"Cleaning up temporary directory: {EXTRACT_DIR_TEMP}")
    try:
        # 如果移动成功，EXTRACT_DIR_TEMP 应该是空的或只包含原顶层目录（如果移动的是子目录）
        shutil.rmtree(EXTRACT_DIR_TEMP)
    except OSError as e:
         print(f"Warning: Could not completely remove temp directory {EXTRACT_DIR_TEMP}: {e}")


    print("\nONNX Runtime dependency setup complete.")

if __name__ == "__main__":
    main()