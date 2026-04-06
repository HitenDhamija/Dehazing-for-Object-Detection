import os
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[0]

# Model run mapping
models_map = {
    'AOD-Net': 'runs/SIR_AOD_Dehazed_yolo/labels',
    'FFA-Net': 'runs/SIR_FFA_Dehazed_yolo/labels',
    'FFA-Net Enhanced': 'runs/SIR_FFA_Enhanced_yolo/labels',
    'DCP': 'runs/detect/sir_dcp_refined/labels',
    'GridDehazeNet1 (Pre-DCP)': 'runs/gridDehazeResults/labels',
}

results = []

for model_name, label_dir in models_map.items():
    label_path = ROOT / label_dir
    
    if not label_path.exists():
        print(f"❌ {model_name}: Path not found - {label_path}")
        results.append({
            'Model': model_name,
            'Total Detections': 'N/A',
            'Avg Objects/Image': 'N/A',
            'Total Images': 'N/A',
            'Detection Rate': 'N/A',
            'Unique Classes': 'N/A'
        })
        continue
    
    label_files = list(label_path.glob('*.txt'))
    if not label_files:
        print(f"❌ {model_name}: No label files found")
        results.append({
            'Model': model_name,
            'Total Detections': 0,
            'Avg Objects/Image': 0,
            'Total Images': 0,
            'Detection Rate': 0,
            'Unique Classes': 0
        })
        continue
    
    total_detections = 0
    total_images = len(label_files)
    classes_set = set()
    detections_per_image = []
    
    for label_file in sorted(label_files):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                detection_count = len(lines)
                total_detections += detection_count
                detections_per_image.append(detection_count)
                
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        classes_set.add(parts[0])  # class ID
        except:
            pass
    
    avg_objects = total_detections / total_images if total_images > 0 else 0
    unique_classes = len(classes_set)
    detection_rate = (total_detections / (total_images * 5)) * 100 if total_images > 0 else 0  # Assume max 5 objects per image
    
    results.append({
        'Model': model_name,
        'Total Detections': total_detections,
        'Avg Objects/Image': round(avg_objects, 2),
        'Total Images': total_images,
        'Detection Rate (%)': round(detection_rate, 1),
        'Unique Classes': unique_classes
    })
    
    print(f"✅ {model_name}: {total_detections} detections across {total_images} images")

# Create table
print("\n" + "="*120)
print("MODEL COMPARISON TABLE")
print("="*120)

headers = ['Model', 'Total Detections', 'Avg Objects/Image', 'Total Images', 'Detection Rate (%)', 'Unique Classes']
col_widths = [30, 20, 20, 18, 20, 18]

# Print header
header_str = ""
for i, h in enumerate(headers):
    header_str += h.ljust(col_widths[i])
print(header_str)
print("-" * 120)

# Print rows
for result in results:
    row_str = ""
    for i, h in enumerate(headers):
        val = str(result.get(h, 'N/A'))
        row_str += val.ljust(col_widths[i])
    print(row_str)

print("="*120)

# Save to CSV
output_csv = ROOT / 'model_comparison.csv'
csv_fieldnames = ['Model', 'Total Detections', 'Avg Objects/Image', 'Total Images', 'Detection Rate (%)', 'Unique Classes']
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Results saved to: {output_csv}")

# Summary
print("\n" + "="*120)
print("PERFORMCANCE SUMMARY - Ranked by Total Detections")
print("="*120)

# Sort by total detections
sorted_results = sorted([r for r in results if r['Total Detections'] != 'N/A'], 
                       key=lambda x: int(x['Total Detections']), reverse=True)

for rank, result in enumerate(sorted_results, 1):
    detections = int(result['Total Detections'])
    avg_obj = result['Avg Objects/Image']
    classes = result['Unique Classes']
    print(f"{rank}. {result['Model']:<30} - {detections:>3} detections | Avg {avg_obj}/img | {classes} classes")

print("="*120)
