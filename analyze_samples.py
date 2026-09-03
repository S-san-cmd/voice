import os
import sys
from feature_extractor import extract_features
from rule_classifier import classify_voice

folder = r"c:\AntiGravity\Onnagoe\声変わりこなかった成人男性"
files = [f for f in os.listdir(folder) if f.endswith(('.mp3', '.m4a', '.wav'))]

for f in files:
    path = os.path.join(folder, f)
    print(f"--- Analyzing {f} ---")
    features, err = extract_features(path)
    if err:
        print(f"Error: {err}")
        continue
    
    res, reasons = classify_voice(features)
    
    print(f"F0: {features.get('f0_median', 0):.1f} Hz")
    print(f"F1: {features.get('f1_median', 0):.1f} Hz")
    print(f"F2: {features.get('f2_median', 0):.1f} Hz")
    print(f"F3: {features.get('f3_median', 0):.1f} Hz")
    print(f"F4: {features.get('f4_median', 0):.1f} Hz")
    print(f"VTL: {features.get('vtl', 0):.2f} cm")
    print(f"HNR: {features.get('hnr_median', 0):.1f} dB")
    print(f"Jitter: {features.get('jitter', 0):.4f}")
    print(f"Shimmer: {features.get('shimmer', 0):.4f}")
    print(f"Result: {res}")
    for r in reasons:
        print(f" - {r}")
    print()
