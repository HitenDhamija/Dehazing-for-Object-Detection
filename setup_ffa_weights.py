import os
import sys

# Add model download script
def download_ffa_weights():
    """Download FFA-Net pre-trained weights from available sources"""
    import requests
    from pathlib import Path
    
    output_dir = Path("models/FFA-Net/net/trained_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try alternative sources
    print("Attempting to download FFA-Net pre-trained weights...")
    
    # Model info
    model_name = "its_train_ffa_3_19.pk"
    model_path = output_dir / model_name
    
    if model_path.exists():
        print(f"✅ Model already exists at {model_path}")
        return True
    
    print("\n⚠️ FFA-Net pre-trained weights not found!")
    print("\nTo get the proper dehazing results, download weights from:")
    print("📥 Google Drive: https://drive.google.com/drive/folders/19_lSUPrpLDZl9AyewhHBsHidZEpTMIV5")
    print("   - Download 'its_train_ffa_3_19.pk' file")
    print("   - Place it in: models/FFA-Net/net/trained_models/")
    print("\n📥 Or Baidu Pan: https://pan.baidu.com/s/1-pgSXN6-NXLzmTp21L_qIg (code: 4gat)")
    print("   - Download the 'its_train_ffa_3_19.pk' model")
    print("   - Place it in: models/FFA-Net/net/trained_models/")
    
    return False

if __name__ == "__main__":
    success = download_ffa_weights()
    sys.exit(0 if success else 1)