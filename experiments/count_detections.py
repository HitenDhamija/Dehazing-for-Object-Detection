import os

def count(folder):
    total = 0
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file)) as f:
                total += len(f.readlines())
    return total

baseline = "runs/detect/baseline_hazy/labels"
dehazed = "runs/detect/ffa_dehazed/labels"

print("Baseline detections:", count(baseline))
print("Dehazed detections:", count(dehazed))