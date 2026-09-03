import os
import joblib
import numpy as np

MODEL_PATH = r"c:\AntiGravity\Onnagoe\model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
            return None
    return None

def predict_voice(features, model):
    from rule_classifier import estimate_vtl
    vtl = estimate_vtl(features.get('f1_median', 0), 
                       features.get('f2_median', 0), 
                       features.get('f3_median', 0), 
                       features.get('f4_median', 0))
                       
    row = [
        features.get('f0_median', 0),
        features.get('f1_median', 0),
        features.get('f2_median', 0),
        features.get('f3_median', 0),
        features.get('f4_median', 0),
        features.get('f0_std', 0),
        features.get('f1_std', 0),
        features.get('f2_std', 0),
        features.get('f3_std', 0),
        features.get('f4_std', 0),
        vtl,
        features.get('hnr_median', 0),
        features.get('jitter', 0),
        features.get('shimmer', 0),
        features.get('h1_db', 0.0),
        features.get('a1_db', 0.0),
        features.get('a2_db', 0.0),
        features.get('a3_db', 0.0),
        features.get('h1_a1', 0.0),
        features.get('h1_a3', 0.0)
    ]
    
    mfcc = features.get('mfcc_mean', [])
    if len(mfcc) < 20:
        mfcc = mfcc + [0]*(20-len(mfcc))
    row.extend(mfcc)

    mfcc_s = features.get('mfcc_std', [])
    if len(mfcc_s) < 20:
        mfcc_s = mfcc_s + [0]*(20-len(mfcc_s))
    row.extend(mfcc_s)
    
    X = np.array([row])
    
    pred_class = model.predict(X)[0]
    
    try:
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
        prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
    except:
        prob_dict = {}
        
    return pred_class, prob_dict
