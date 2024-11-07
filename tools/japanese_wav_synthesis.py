import re
import os
import argparse
import time
import openai
import requests

# 设置OpenAI API密钥
# openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = ""
api_base_url = ""

def synthesize_speech(text, filename, retries=3):
    attempt = 0
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "voice": "ja-JP-Standard-A",
        "response_format": "audio/wav"
    }

    while attempt < retries:
        try:
            # 使用OpenAI API进行语音合成请求
            response = requests.post(f"{api_base_url}/v1/audio/synthesize", headers=headers, json=data, verify=False)
            response.raise_for_status()
            with open(filename, "wb") as audio_file:
                audio_file.write(response.content)
            break
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                print(f"Failed to generate speech for '{text}': {e}")
            else:
                print(f"Retrying ({attempt}/{retries})...")
                time.sleep(2)

def add_audio_link(line, audio_filename):
    return f'{line}\n<audio controls><source src="{audio_filename}" type="audio/mpeg"></audio>\n'

def process_markdown(input_file, output_file, audio_dir='audio'):
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            # 检查行中是否有日语
            if re.search(r'[\u3040-\u30ff\u4e00-\u9faf]', line):
                # 如果包含日语，生成音频文件
                audio_filename = os.path.join(audio_dir, f"audio_{hash(line)}.wav")
                synthesize_speech(line.strip(), audio_filename)
                # 将音频链接加入到 markdown 中
                line = add_audio_link(line, audio_filename)
            
            f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a Markdown file and generate audio for Japanese text.")
    parser.add_argument('--input_file', type=str, default="/root/blog/jxchen-blog/docs/japanese/gramma/class_1.md", help="Path to the input Markdown file.")
    parser.add_argument('--output_file', type=str, default="/root/blog/jxchen-blog/docs/japanese/gramma/class_1_with_wav.md", help="Path to the output Markdown file.")
    parser.add_argument('--audio_dir', type=str, default="/root/blog/jxchen-blog/docs/japanese/wav", help="Directory to save audio files.")
    
    args = parser.parse_args()

    process_markdown(args.input_file, args.output_file, args.audio_dir)