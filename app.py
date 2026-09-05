import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import importlib
import urllib.parse

import feature_extractor
import rule_classifier
import ml_classifier
importlib.reload(feature_extractor)
importlib.reload(rule_classifier)
importlib.reload(ml_classifier)

from feature_extractor import extract_features
from rule_classifier import classify_voice, evaluate_voice_quality
from ml_classifier import load_model, predict_voice

st.set_page_config(page_title="発声タイプ判定", layout="wide")
st.title("発声タイプ判定")
st.write("WAVファイルまたはMP3ファイルをアップロードして、声の音響的特徴からジェンダーや発声タイプを推定します。")

app_mode = st.radio("表示モードを選択してください", ["一般向け", "両声類向け"])

input_method = st.radio("音声の入力方法を選択してください", ["ファイルアップロード", "マイクで録音（5秒以上）"])

audio_source = None
if input_method == "ファイルアップロード":
    audio_source = st.file_uploader("音声ファイルを選択してください", type=["wav", "mp3", "m4a"])
else:
    audio_source = st.audio_input("マイクから音声を録音してください（5秒以上）")
    st.write(
        "吾輩（わがはい）は猫である。名前はまだ無い。  \n"
        "どこで生れたかとんと見当（けんとう）がつかぬ。  \n"
        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。  \n"
        "吾輩はここで始めて人間というものを見た。"
    )

if audio_source is not None:
    st.audio(audio_source)
    
    if st.button("解析実行"):
        with st.spinner("音声を解析中..."):
            filename = getattr(audio_source, 'name', '.wav')
            ext = os.path.splitext(filename)[1]
            if not ext:
                ext = ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(audio_source.getvalue())
                tmp_path = tmp.name
            
            features, err_msg = extract_features(tmp_path)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            
            if features is None:
                if "短すぎます" in err_msg:
                    st.warning(err_msg)
                else:
                    st.error("解析に失敗しました。音声ファイルを確認してください。")
                    if "NoBackendError" in err_msg or "audioread" in err_msg:
                        st.warning("M4Aファイルの読み込みに必要な 'ffmpeg' がシステムに見つからない可能性があります。")
                    st.code(err_msg)
            else:
                result, reasons = classify_voice(features)
                
                ai_model = load_model()
                app_url = "https://q8nrqzqtgxkukxmyxoxym5.streamlit.app"
                
                if ai_model:
                    def format_class_name(c):
                        if "両声類" in c: return "両声類"
                        if "女性" in c: return "女性の声"
                        if "男性" in c: return "男性の声"
                        return c

                    ai_pred, ai_probs = predict_voice(features, ai_model)
                    display_pred = format_class_name(ai_pred)
                    
                    female_prob = ai_probs.get("シス女性", ai_probs.get("女性", 0)) * 100
                    
                    sorted_probs = sorted(ai_probs.items(), key=lambda item: item[1], reverse=True)
                    top1_class = format_class_name(sorted_probs[0][0])
                    top1_prob = sorted_probs[0][1] * 100
                    top2_class = format_class_name(sorted_probs[1][0]) if len(sorted_probs) > 1 else ""
                    top2_prob = sorted_probs[1][1] * 100 if len(sorted_probs) > 1 else 0
                    
                    if app_mode == "両声類向け":
                        st.markdown(f"<h2 style='text-align: center; color: #ff4b4b;'>女性の可能性: {female_prob:.1f}%</h2>", unsafe_allow_html=True)
                        tweet_text = (f"私の声の「女性の可能性」は {female_prob:.1f}% でした！✨\n\n"
                                      f"あなたも声を測定してみよう！\n#発声タイプ判定\n{app_url}")
                    else:
                        st.markdown(f"<h2 style='text-align: center; color: #4b8bff;'>1位: {top1_class} ({top1_prob:.1f}%)</h2>", unsafe_allow_html=True)
                        if top2_class:
                            st.markdown(f"<h3 style='text-align: center; color: #666666;'>2位: {top2_class} ({top2_prob:.1f}%)</h3>", unsafe_allow_html=True)
                        
                        tweet_text = f"私の声のAI判定結果は 1位: {top1_class}({top1_prob:.1f}%)"
                        if top2_class:
                            tweet_text += f"、2位: {top2_class}({top2_prob:.1f}%)"
                        tweet_text += f" でした！✨\n\nあなたも声を測定してみよう！\n#発声タイプ判定\n{app_url}"

                    st.subheader(f"🤖 AI判定結果: **{display_pred}**")
                    prob_str = ", ".join([f"{format_class_name(c)}: {p*100:.1f}%" for c, p in ai_probs.items()])
                    st.write(f"すべての確信度: {prob_str}")
                else:
                    st.subheader(f"推定結果: {result}")
                    st.info("💡 学習データ（datasetフォルダ）を追加して train_model.py を実行すると、AI判定が有効になります！")
                    
                    for reason in reasons:
                        st.write(f"- {reason}")
                    
                    tweet_text = (f"私の声の推定結果は「{result}」でした！✨\n\n"
                                  f"あなたも声を測定してみよう！\n#発声タイプ判定\n{app_url}")
                
                score, rank, quality_comments = evaluate_voice_quality(features)
                st.markdown("---")
                st.subheader(f"声の魅力度・安定性評価: {rank}")
                st.progress(score / 100)
                st.write(f"**スコア**: {score} / 100")
                for qc in quality_comments:
                    st.write(f"- {qc}")
                
                # X Share Button
                st.markdown("---")
                encoded_tweet = urllib.parse.quote(tweet_text)
                share_url = f"https://twitter.com/intent/tweet?text={encoded_tweet}"
                st.markdown(f'<a href="{share_url}" target="_blank" style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #000000; border-radius: 5px; text-decoration: none; font-weight: bold;">𝕏 で結果をシェアする</a>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("抽出された音響パラメータ")
                    df_features = pd.DataFrame({
                        "パラメータ": [
                            "基本周波数 (F0) Median", 
                            "フォルマント F1 Median", 
                            "フォルマント F2 Median", 
                            "フォルマント F3 Median", 
                            "フォルマント F4 Median", 
                            "推定声道長 (VTL)",
                            "H1-A1 (気息性指標)",
                            "H1-A3 (スペクトル傾斜)",
                            "Harmonics-to-Noise Ratio (HNR)", 
                            "Jitter (局所的ピッチ揺らぎ)", 
                            "Shimmer (局所的振幅揺らぎ)"
                        ],
                        "値": [
                            f"{features['f0_median']:.1f} Hz",
                            f"{features['f1_median']:.1f} Hz",
                            f"{features['f2_median']:.1f} Hz",
                            f"{features['f3_median']:.1f} Hz",
                            f"{features['f4_median']:.1f} Hz",
                            f"{features.get('vtl', 0):.1f} cm",
                            f"{features.get('h1_a1', 0):.1f} dB",
                            f"{features.get('h1_a3', 0):.1f} dB",
                            f"{features['hnr_median']:.1f} dB",
                            f"{features['jitter']:.4f}",
                            f"{features['shimmer']:.4f}"
                        ]
                    })
                    st.table(df_features)
                    
                with col2:
                    st.subheader("F0 vs F2 散布図")
                    st.write("一般的な男女の分布域に対して、解析対象の声がどの位置にあるかを示します。")
                    
                    fig = go.Figure()
                    
                    # 仮想的なシス男性分布
                    fig.add_shape(type="rect",
                        x0=90, y0=1200, x1=160, y1=1600,
                        line=dict(color="blue", width=2),
                        fillcolor="rgba(0,0,255,0.1)",
                        name="男性(目安)"
                    )
                    
                    # 仮想的なシス女性分布
                    fig.add_shape(type="rect",
                        x0=180, y0=1700, x1=250, y1=2200,
                        line=dict(color="red", width=2),
                        fillcolor="rgba(255,0,0,0.1)",
                        name="シス女性(目安)"
                    )
                    
                    # ユーザーのプロット
                    fig.add_trace(go.Scatter(
                        x=[features['f0_median']], 
                        y=[features['f2_median']],
                        mode='markers+text',
                        marker=dict(color='black', size=12, symbol='star'),
                        text=["Your Voice"],
                        textposition="top center",
                        name="解析対象の声"
                    ))
                    
                    fig.update_layout(
                        xaxis_title="基本周波数 F0 (Hz) - ピッチ",
                        yaxis_title="第2フォルマント F2 (Hz) - 響き",
                        xaxis_range=[50, 400],
                        yaxis_range=[1000, 2500],
                    )
                    st.plotly_chart(fig, use_container_width=True)
