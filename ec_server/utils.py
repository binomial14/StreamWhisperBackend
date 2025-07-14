import os

def save_transcript(file_path, segments, end_time):
    completed_texts = []
    for seg in segments:
        if seg['completed']:
            if float(seg['start']) >= end_time:
                completed_texts.append(seg['text'].strip())
                end_time = float(seg['end'])
    
    existing_text = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_text = f.readlines() # remove tmp
        f.close()

    full_text = [l.strip() for l in existing_text[:-1]] + completed_texts

    if segments[-1]['completed']==False:
        full_text.append(segments[-1]['text'].strip())
    else:
        full_text.append("-")

    with open(file_path, "w", encoding="utf-8") as f:
        for line in full_text:
            f.write(line + "\n")
 
    return end_time