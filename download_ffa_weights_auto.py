#!/usr/bin/env python3
"""
Download FFA-Net pre-trained weights from Google Drive
Usage: python download_ffa_weights_auto.py
"""

import os
import sys
from pathlib import Path

def download_from_gdrive():
    """Try to download using gdown"""
    try:
        import gdown
        
        output_dir = Path("models/FFA-Net/net/trained_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Attempting to download FFA-Net weights from Google Drive...")
        
        # GDrive folder link - trying to extract files
        folder_url = "https://drive.google.com/drive/folders/19_lSUPrpLDZl9AyewhHBsHidZEpTMIV5"
        
        try:
            gdown.download_folder(
                url=folder_url,
                output=str(output_dir),
                quiet=False,
                use_cookies=False
            )
            print("✅ Download completed!")
            return True
        except Exception as e:
            print(f"Folder download failed: {e}")
            
            # Try specific file if ID known
            print("\nTrying direct file download...")
            file_urls = {
                "its_train_ffa_3_19.pk": "1rljm2tcHxAp8c8LzVaHhyXZaM7qFyI89",
                "its_train_ffa_3_19_1.pk": "1V9_j5Yk5wN5ZfW9RR-8Wn7xdZE5wW5x5",
            }
            
            for filename, file_id in file_urls.items():
                try:
                    print(f"Downloading {filename}...")
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_id}",
                        output=str(output_dir / filename),
                        quiet=False
                    )
                    print(f"✅ Downloaded {filename}")
                    return True
                except:
                    pass
            
            return False
            
    except ImportError:
        print("gdown not installed")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def manual_download_instructions():
    """Show manual download instructions"""
    print("\n" + "="*70)
    print("📥 MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\n1. Google Drive Option:")
    print("   URL: https://drive.google.com/drive/folders/19_lSUPrpLDZl9AyewhHBsHidZEpTMIV5")
    print("   - Download 'its_train_ffa_3_19.pk'")
    print("   - Save to: models/FFA-Net/net/trained_models/")
    
    print("\n2. Baidu Pan Option:")
    print("   URL: https://pan.baidu.com/s/1-pgSXN6-NXLzmTp21L_qIg")
    print("   Password: 4gat")
    print("   - Download 'its_train_ffa_3_19.pk'")
    print("   - Save to: models/FFA-Net/net/trained_models/")
    
    print("\n3. After downloading, run:")
    print("   python experiments/ffanet_yolo.py")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print("FFA-Net Weight Downloader")
    print("="*70)
    
    weights_dir = Path("models/FFA-Net/net/trained_models")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if weights already exist
    for ext in ['.pk', '.pth']:
        existing = list(weights_dir.glob(f"*ffa*{ext}"))
        if existing:
            print(f"✅ Found existing weights: {existing[0]}")
            sys.exit(0)
    
    print("\nAttempting automatic download...")
    if download_from_gdrive():
        print("✅ Weights downloaded successfully!")
        sys.exit(0)
    else:
        print("❌ Automatic download failed")
        manual_download_instructions()
        sys.exit(1)
