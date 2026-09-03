import parselmouth
from parselmouth.praat import call
import librosa
import numpy as np
import os
import tempfile
import traceback
import subprocess

try:
    import imageio_ffmpeg
except ImportError:
    pass

def extract_features(filepath):
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"
        
    try:
        # ffmpegを直接呼び出してあらゆる音声形式を確実にWAVに変換する（ffprobe不足によるエラーを回避）
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Praat(Parselmouth)で確実に処理できるよう、モノラル・16bitPCMのWAVファイルとして一時保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            clean_wav_path = tmp_wav.name
            
        cmd = [ffmpeg_exe, "-y", "-i", filepath, "-acodec", "pcm_s16le", "-ac", "1", clean_wav_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Load sound via praat
        sound = parselmouth.Sound(clean_wav_path)
        
        # Check audio duration
        duration = sound.get_total_duration()
        if duration < 5.0:
            os.remove(clean_wav_path)
            return None, f"音声が短すぎます（{duration:.1f}秒）。正確な解析のため、5秒以上の音声を入力してください。"

        
        # Pitch (F0)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        f0_median = np.median(pitch_values) if len(pitch_values) > 0 else 0.0
        f0_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
        
        # Formants
        # Setting maximum_formant=5500 for female, 5000 for male. 
        # Using 5500 as a safer bet for high voices.
        formant = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)
        
        f1_list, f2_list, f3_list, f4_list = [], [], [], []
        # Sample formants at pitch points
        for t in pitch.xs():
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            f4 = formant.get_value_at_time(4, t)
            if not np.isnan(f1): f1_list.append(f1)
            if not np.isnan(f2): f2_list.append(f2)
            if not np.isnan(f3): f3_list.append(f3)
            if not np.isnan(f4): f4_list.append(f4)
            
        f1_median = np.median(f1_list) if f1_list else 0.0
        f2_median = np.median(f2_list) if f2_list else 0.0
        f3_median = np.median(f3_list) if f3_list else 0.0
        f4_median = np.median(f4_list) if f4_list else 0.0

        f0_std = np.std(pitch_values) if len(pitch_values) > 0 else 0.0
        f1_std = np.std(f1_list) if f1_list else 0.0
        f2_std = np.std(f2_list) if f2_list else 0.0
        f3_std = np.std(f3_list) if f3_list else 0.0
        f4_std = np.std(f4_list) if f4_list else 0.0
        
        # Harmonicity (HNR)
        harmonicity = sound.to_harmonicity_cc()
        hnr_values = harmonicity.values
        hnr_values = hnr_values[hnr_values != -200]
        hnr_median = np.median(hnr_values) if len(hnr_values) > 0 else 0.0
        
        # Jitter and Shimmer
        pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Jitter/Shimmer can return string "undefined" if no periods found
        if isinstance(localJitter, str): localJitter = 0.0
        if isinstance(localShimmer, str): localShimmer = 0.0
        
        # MFCC (using librosa directly from the clean WAV file)
        y, sr = librosa.load(clean_wav_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1).tolist()
        mfcc_std = np.std(mfccs, axis=1).tolist()
        
        # LTAS (Long Term Average Spectrum) for amplitude features
        try:
            ltas = call(sound, "To Ltas", 1) # Bin size 1 Hz
            def get_peak_db(freq):
                if freq <= 0: return 0.0
                f_low = freq * 0.8
                f_high = freq * 1.2
                try:
                    val = call(ltas, "Get maximum", f_low, f_high, "None")
                    return float(val) if not np.isnan(val) and not np.isinf(val) else 0.0
                except:
                    return 0.0
            
            h1_db = get_peak_db(f0_median)
            a1_db = get_peak_db(f1_median)
            a2_db = get_peak_db(f2_median)
            a3_db = get_peak_db(f3_median)
        except:
            h1_db, a1_db, a2_db, a3_db = 0.0, 0.0, 0.0, 0.0

        # 一時作成したWAVファイルの削除
        os.remove(clean_wav_path)
        
        features = {
            "f0_mean": float(f0_mean),
            "f0_median": float(f0_median),
            "f1_median": float(f1_median),
            "f2_median": float(f2_median),
            "f3_median": float(f3_median),
            "f4_median": float(f4_median),
            "f0_std": float(f0_std),
            "f1_std": float(f1_std),
            "f2_std": float(f2_std),
            "f3_std": float(f3_std),
            "f4_std": float(f4_std),
            "hnr_median": float(hnr_median),
            "jitter": float(localJitter),
            "shimmer": float(localShimmer),
            "h1_db": float(h1_db),
            "a1_db": float(a1_db),
            "a2_db": float(a2_db),
            "a3_db": float(a3_db),
            "h1_a1": float(h1_db - a1_db) if h1_db != 0.0 and a1_db != 0.0 else 0.0,
            "h1_a3": float(h1_db - a3_db) if h1_db != 0.0 and a3_db != 0.0 else 0.0,
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std
        }

        return features, None
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Error processing {filepath}:\n{err_msg}")
        if 'clean_wav_path' in locals() and os.path.exists(clean_wav_path):
            try:
                os.remove(clean_wav_path)
            except:
                pass
        return None, err_msg
