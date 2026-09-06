import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import importlib
import urllib.parse
import io
from PIL import Image, ImageDraw, ImageFont
import feature_extractor
import rule_classifier
import ml_classifier


from feature_extractor import extract_features
from rule_classifier import classify_voice, evaluate_voice_quality
from ml_classifier import load_model, predict_voice
import gc

@st.cache_resource
def get_cached_model():
    return load_model()


def generate_result_image(app_mode, top1_class, top1_prob, top2_class, top2_prob, female_prob, voice_score=None):
    width, height = 1200, 630
    img = Image.new('RGB', (width, height), color='#f0f4f8')
    draw = ImageDraw.Draw(img)
    
    font_path = "font.ttf"
    try:
        font_xlarge = ImageFont.truetype(font_path, 110)
        font_large = ImageFont.truetype(font_path, 80)
        font_medium = ImageFont.truetype(font_path, 50)
        font_small = ImageFont.truetype(font_path, 35)
    except Exception:
        font_xlarge = ImageFont.load_default()
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
    draw.text((width/2, 80), "発声タイプ判定結果", font=font_medium, fill='#333333', anchor="mm")
    
    if app_mode == "男性両声類向け":
        if voice_score is not None:
            draw.text((width/2, 220), "女声スコア", font=font_medium, fill='#ff4b4b', anchor="mm")
            draw.text((width/2, 330), f"{voice_score:.1f}", font=font_xlarge, fill='#ff4b4b', anchor="mm")
        draw.text((width/2, 450), f"女性の可能性: {female_prob:.1f}%", font=font_small, fill='#333333', anchor="mm")
        if top1_class:
            draw.text((width/2, 510), f"({top1_class} {top1_prob:.1f}%)", font=font_small, fill='#666666', anchor="mm")
    else:
        draw.text((width/2, 300), f"1位: {top1_class} ({top1_prob:.1f}%)", font=font_large, fill='#4b8bff', anchor="mm")
        if top2_class:
            draw.text((width/2, 430), f"2位: {top2_class} ({top2_prob:.1f}%)", font=font_medium, fill='#666666', anchor="mm")
            
    draw.text((width/2, 570), "あなたも声を測定してみよう！ #発声タイプ判定", font=font_small, fill='#888888', anchor="mm")
    
    draw.rectangle([20, 20, width-20, height-20], outline="#cccccc", width=8)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

st.set_page_config(page_title="発声タイプ判定", layout="wide")
st.title("発声タイプ判定")
st.write("WAVファイルまたはMP3ファイルをアップロードして、声の音響的特徴からジェンダーや発声タイプを推定します。")

app_mode = st.radio("表示モードを選択してください", ["一般向け", "男性両声類向け"])

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
                
                ai_model = get_cached_model()
                app_url = "https://q8nrqzqtgxkukxmyxoxym5.streamlit.app"
                
                # メモリ解放
                gc.collect()
                
                if ai_model:
                    def format_class_name(c):
                        if "両声類" in c: return "両声類"
                        if "女性" in c: return "女性の声"
                        if "未変声" in c: return "未変声の男性の声"
                        if "裏声" in c: return "男性の裏声"
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
                    
                    if app_mode == "男性両声類向け":
                        prob_female = 0
                        prob_ryou = 0
                        prob_mihen = 0
                        prob_uragoe = 0
                        
                        for k, v in ai_probs.items():
                            if "女性" in k:
                                prob_female += v * 100
                            elif "両声類" in k:
                                prob_ryou += v * 100
                            elif "未変声" in k:
                                prob_mihen += v * 100
                            elif "裏声" in k:
                                prob_uragoe += v * 100
                                
                        voice_score = prob_female + (prob_ryou / 2) + (prob_mihen * 0.7) - prob_uragoe
                        
                        non_female_probs = [item for item in sorted_probs if "女性" not in format_class_name(item[0])]
                        if non_female_probs:
                            top_nf_class = format_class_name(non_female_probs[0][0])
                            top_nf_prob = non_female_probs[0][1] * 100
                        else:
                            top_nf_class = ""
                            top_nf_prob = 0
                            
                        st.markdown(f"<h1 style='text-align: center; color: #ff4b4b; font-size: 3em;'>女声スコア: {voice_score:.1f}</h1>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='text-align: center; color: #333333;'>女性の可能性: {female_prob:.1f}%</h3>", unsafe_allow_html=True)
                        if top_nf_class:
                            st.markdown(f"<h4 style='text-align: center; color: #666666;'>({top_nf_class} {top_nf_prob:.1f}%)</h4>", unsafe_allow_html=True)
                            
                        tweet_text = f"私の「女声スコア」は {voice_score:.1f} でした！✨\n"
                        tweet_text += f"女性の可能性: {female_prob:.1f}%\n"
                        if top_nf_class:
                            tweet_text += f"（{top_nf_class} {top_nf_prob:.1f}%）\n\n"
                        else:
                            tweet_text += "\n"
                        tweet_text += f"あなたも声を測定してみよう！\n#発声タイプ判定\n{app_url}"
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
                
                # シェア用画像の生成と表示
                st.markdown("---")
                st.subheader("シェア用画像")
                st.write("この画像を保存（長押し or 右クリック）して、X（Twitter）の投稿に添付してください。")
                
                if app_mode == "男性両声類向け" and 'top_nf_class' in locals():
                    t1_class = top_nf_class
                    t1_prob = top_nf_prob
                else:
                    t1_class = top1_class if 'top1_class' in locals() else result
                    t1_prob = top1_prob if 'top1_prob' in locals() else 100.0
                    
                t2_class = top2_class if 'top2_class' in locals() else ""
                t2_prob = top2_prob if 'top2_prob' in locals() else 0.0
                f_prob = female_prob if 'female_prob' in locals() else 0.0
                
                v_score = voice_score if 'voice_score' in locals() else None
                
                result_img_bytes = generate_result_image(app_mode, t1_class, t1_prob, t2_class, t2_prob, f_prob, voice_score=v_score)
                st.image(result_img_bytes, use_container_width=True)
                st.download_button(
                    label="📷 画像をダウンロードする",
                    data=result_img_bytes,
                    file_name="voice_result.png",
                    mime="image/png"
                )
                
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
