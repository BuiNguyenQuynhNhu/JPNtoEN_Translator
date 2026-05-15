import os
import re
import math
import argparse
from tqdm import tqdm

def srt_time_to_ms(time_str):
    """Converts SRT timestamp HH:MM:SS,mmm to milliseconds."""
    h, m, s_mmm = time_str.split(':')
    s, mmm = s_mmm.split(',')
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(mmm)

def ms_to_srt_time(ms):
    """Converts milliseconds back to SRT timestamp format HH:MM:SS,mmm."""
    ms = max(0, int(ms))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    mmm = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{mmm:03d}"

def clean_text(text):
    """Trim whitespace and remove weird invisible characters."""
    if not text: return ""
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF\ufffc\xad]', '', text)
    return text.strip()

def parse_srt(filepath):
    """Parses SRT file and extracts timestamps into milliseconds for calculation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = content.strip().split('\n\n')
    parsed_blocks = []
    
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 2:
            index = lines[0]
            timestamp = lines[1]
            text = '\n'.join(lines[2:]) if len(lines) > 2 else ""
            
            if '-->' in timestamp:
                start_str, end_str = timestamp.split('-->')
                start_ms = srt_time_to_ms(start_str.strip())
                end_ms = srt_time_to_ms(end_str.strip())
                
                parsed_blocks.append({
                    'index': int(index) if index.isdigit() else 0,
                    'timestamp': timestamp,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'text': text,
                    'original_text': text
                })
                
    parsed_blocks.sort(key=lambda x: x['index'])
    return parsed_blocks

def adjust_timestamps(blocks, reading_speed_cps=20, min_gap_ms=100):
    """
    Dynamically extends the duration of subtitles based on English reading speed.
    Ensures that extended subtitles do not overlap the next subtitle.
    """
    for i in range(len(blocks)):
        block = blocks[i]
        
        # Calculate required milliseconds based on characters per second (CPS)
        text_len = len(block['text'].replace('\n', ' '))
        required_ms = int((text_len / reading_speed_cps) * 1000)
        current_ms = block['end_ms'] - block['start_ms']
        
        if required_ms > current_ms:
            new_end_ms = block['start_ms'] + required_ms
            
            # Boundary constraint: Cannot overlap the next subtitle minus a small gap
            if i + 1 < len(blocks):
                next_start_ms = blocks[i+1]['start_ms']
                max_allowed_end = next_start_ms - min_gap_ms
                if new_end_ms > max_allowed_end:
                    new_end_ms = max_allowed_end
            
            # Only update if the new end is actually longer than the current end
            if new_end_ms > block['end_ms']:
                block['end_ms'] = new_end_ms
                
        # Reconstruct standard timestamp string
        block['timestamp'] = f"{ms_to_srt_time(block['start_ms'])} --> {ms_to_srt_time(block['end_ms'])}"
        
    return blocks

def merge_and_adjust(original_blocks, translations, cps=20):
    merged_srt = []
    cleaned_translations = [clean_text(t) for t in translations]
    
    num_orig = len(original_blocks)
    num_trans = len(cleaned_translations)
    
    for i in range(num_orig):
        block = original_blocks[i].copy()
        
        if i < num_trans:
            block['text'] = cleaned_translations[i]
        else:
            block['text'] = ""
            
        block['index'] = i + 1
        merged_srt.append(block)
        
    # Apply Smart Retiming
    merged_srt = adjust_timestamps(merged_srt, reading_speed_cps=cps)
    return merged_srt

def write_srt(srt_blocks, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for block in srt_blocks:
            f.write(f"{block['index']}\n")
            f.write(f"{block['timestamp']}\n")
            if block['text']:
                f.write(f"{block['text']}\n")
            else:
                f.write(" \n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="End-to-End SRT Translation with Smart Retiming.")
    parser.add_argument("-i", "--input", required=True, help="Path to original Japanese SRT file")
    parser.add_argument("-o", "--output", required=True, help="Path to save translated English SRT file")
    parser.add_argument("--cps", type=int, default=20, help="Reading Speed (Characters Per Second) to adjust duration")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for model translation")
    args = parser.parse_args()
    
    print(f"Parsing SRT file: {args.input}")
    blocks = parse_srt(args.input)
    japanese_texts = [b['text'].replace('\n', ' ') for b in blocks]
    
    print("Loading AI Translation Pipeline...")
    # Import inside to prevent slow loads if just showing help
    from src.models.inference import TranslationPipeline
    pipeline = TranslationPipeline()
    
    print(f"Translating {len(japanese_texts)} subtitle blocks...")
    translated_texts = []
    
    # Process in batches for progress bar and memory efficiency
    for i in tqdm(range(0, len(japanese_texts), args.batch), desc="Translating"):
        batch_texts = japanese_texts[i : i + args.batch]
        results = pipeline.translate_subtitles(batch_texts)
        translated_texts.extend(results)
        
    print("Merging translations and applying Smart Retiming...")
    final_blocks = merge_and_adjust(blocks, translated_texts, cps=args.cps)
    
    write_srt(final_blocks, args.output)
    print(f"Done! Translated subtitle saved to: {args.output}")

if __name__ == "__main__":
    main()
