import numpy as np

def estimate_vtl(f1, f2, f3, f4):
    if f1 == 0 or f2 == 0 or f3 == 0 or f4 == 0:
        return 0.0
    delta_f = (0.5*f1 + 1.5*f2 + 2.5*f3 + 3.5*f4) / 21.0
    if delta_f == 0:
        return 0.0
    return 35000 / (2 * delta_f)

def classify_voice(features):
    if not features:
        return "判定不能", ["音声の解析に失敗しました。"]

    f0 = features.get('f0_median', 0)
    f1 = features.get("f1_median", 0)
    f2 = features.get('f2_median', 0)
    f3 = features.get('f3_median', 0)
    f4 = features.get('f4_median', 0)
    hnr = features.get('hnr_median', 0)
    jitter = features.get('jitter', 0)
    shimmer = features.get('shimmer', 0)
    
    vtl = estimate_vtl(f1, f2, f3, f4)

    result = "判定不能"
    reasons = []

    if f0 == 0 or f2 == 0:
        return "音声なし/解析失敗", ["有声音（ピッチやフォルマント）が検出されませんでした。"]

    # --- 基本的な閾値 (一般的な成人の目安) ---
    # F0: 男 ~120Hz, 女 ~200Hz-220Hz. 中間 ~160Hz
    # F2: 男 ~1400-1600Hz, 女 ~1700-1900Hz+
    # ------------------------------------

    h1_a1 = features.get('h1_a1', 0)
    h1_a3 = features.get('h1_a3', 0)

    # --- 基本的な閾値 (一般的な成人の目安) ---
    # F0: 男 ~120Hz, 女 ~200Hz-220Hz. 中間 ~160Hz
    # F2: 男 ~1400-1600Hz, 女 ~1700-1900Hz+
    # ------------------------------------

    if f0 < 165:
        # 男性域のピッチ
        result = "男性（地声）"
        reasons.append(f"基本周波数(F0)が男性域（{f0:.1f}Hz）です。")
        if hnr < 10:
            reasons.append(f"気音成分がかなり多く、かすれ声の可能性があります（HNR={hnr:.1f}dB）。")
        if h1_a3 < 15 and h1_a3 != 0:
            reasons.append(f"高音域まで倍音が豊かで（スペクトル傾斜が緩やか: H1-A3={h1_a3:.1f}dB）、男性地声特有の芯のある声質です。")
    else:
        # 女性域または高音域のピッチ
        if f2 < 1650:
            # フォルマントが低い（声道が長い＝男性的な骨格）
            if h1_a3 > 16.0 or hnr < 12.0:
                result = "男性の裏声"
                reasons.append(f"ピッチ(F0)は高い（{f0:.1f}Hz）ですが、フォルマントF2（{f2:.1f}Hz）が低く、スペクトル傾斜や気息性（H1-A3={h1_a3:.1f}dB, HNR={hnr:.1f}dB）からファルセット（男性の裏声）と判定されました。")
            else:
                result = "男性（両声類）"
                reasons.append(f"ピッチ(F0: {f0:.1f}Hz)は女性域ですが、フォルマントF2（{f2:.1f}Hz）に男性的な響きの残る両声類アプローチの特性が見られます。")
        else:
            # フォルマントも高いが、男性骨格の特徴が残っている場合は厳しく判定する
            if vtl >= 15.5:
                 if h1_a3 > 18.0:
                     result = "男性の裏声"
                     reasons.append(f"ピッチやフォルマントは高音域ですが、推定声道長（{vtl:.1f}cm）の長さと高周波の強い減衰（H1-A3={h1_a3:.1f}dB）から、ファルセット（裏声）発声である可能性が高いです。")
                 else:
                     result = "男性（両声類）"
                     reasons.append(f"ピッチ・フォルマント共に女性域にコントロールされていますが、推定声道長（{vtl:.1f}cm）が長く大人の男性骨格を保ったまま女声発声を行う男性両声類としての特徴に合致しています。")
            else:
                 if vtl >= 14.8 and h1_a1 < 3.0:
                     result = "未変声の成人男性"
                     reasons.append(f"ピッチとフォルマントは高いですが、推定声道長（{vtl:.1f}cm）が女性の平均よりもやや長く、気息性（H1-A1={h1_a1:.1f}dB）も低いため、声変わりを経ていない成人男性特有の響きと判断されます。")
                     reasons.append("（※一部の女性の声もこの分類に含まれる場合があります）")
                 else:
                     result = "女性"
                     reasons.append(f"ピッチ(F0: {f0:.1f}Hz)とフォルマント(F2: {f2:.1f}Hz)の両方が高く、推定声道長（{vtl:.1f}cm）も短いため、女性的な音響特性に合致します。")
                     if h1_a1 >= 2.0:
                         reasons.append(f"H1-A1（{h1_a1:.1f}dB）が高く、声帯の後部に隙間がある女性特有の息の混ざった柔らかい声質（気息性）が確認できます。")

    # features辞書にvtlを追加して返す（app.py側で表示するため）
    features["vtl"] = vtl
    return result, reasons

def evaluate_voice_quality(features):
    if not features:
        return 0, "判定不能", ["評価できません。"]
    
    hnr = features.get('hnr_median', 0)
    jitter = features.get('jitter', 0)
    shimmer = features.get('shimmer', 0)
    
    score = 100
    comments = []
    
    # HNR (Harmonics-to-Noise Ratio): 声の澄み具合、ノイズの少なさ
    if hnr > 20:
        comments.append("声の響きが非常にクリアで、芯のある通る声（HNR高）です。")
    elif hnr > 14:
        comments.append("適度なクリアさを持つ標準的な声質です。")
        score -= 10
    elif hnr > 8:
        comments.append("少し息漏れやハスキーさ（かすれ）が混じる声質です（Breathyな魅力とも取れますが物理的にはノイズを含みます）。")
        score -= 20
    else:
        comments.append("声帯の閉鎖が弱く、ノイズ（息漏れ・かすれ）が非常に強い状態です。")
        score -= 40
        
    # Jitter (ピッチの揺らぎ): 声帯振動の安定性
    if jitter < 0.008:
        comments.append("ピッチが極めて安定しており、発声のコントロールが洗練されています。")
    elif jitter < 0.015:
        pass # 標準
        score -= 5
    elif jitter < 0.025:
        comments.append("ピッチに僅かな揺らぎがあり、発声時に少し力みがある可能性があります。")
        score -= 15
    else:
        comments.append("ピッチの揺らぎが大きく、発声が不安定（震え・過度な緊張）です。")
        score -= 30
        
    # Shimmer (振幅の揺らぎ): 声の大きさの安定性
    if shimmer < 0.04:
        comments.append("声のボリュームが一定に保たれており、非常に滑らかです。")
    elif shimmer < 0.08:
        pass # 標準
        score -= 5
    elif shimmer < 0.12:
        comments.append("声のボリュームに少しバラつきがあります。")
        score -= 10
    else:
        comments.append("声のボリュームの揺らぎが大きく、息のコントロールが乱れているか無理な発声をしています。")
        score -= 20
        
    # 総合スコアの補正
    score = int(max(0, min(100, score)))
    
    # 評価ランク
    if score >= 90:
        rank = "S (非常に魅力的・プロレベルの安定した発声)"
    elif score >= 75:
        rank = "A (魅力的・十分安定した声質)"
    elif score >= 60:
        rank = "B (一般的な声質・平均的)"
    elif score >= 40:
        rank = "C (発声にやや不安定さやノイズが目立つ状態)"
    else:
        rank = "D (発声に強い無理や負担がかかっている状態)"
        
    return score, rank, comments

