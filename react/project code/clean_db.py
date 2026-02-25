import json
import re
import os
import glob

DB_PATH = 'backend/data/notes_db.json'

with open(DB_PATH, 'r') as f:
    db = json.load(f)

patterns_to_remove = [
    r'(?i)The average material cost is 64%.*?profit, etc\.?',
    r'(?i)The average material cost is 64%.*?overhead and profit, etc\.',
    r'(?i)\*\*Average Material Cost:\*\* 64%.*?20% of 64%\)',
    r'(?i)Average Material Cost: 64%.*?20% of 64%\)',
    r'(?i)The average material cost is 64%[^\\n]*',
]

def clean_text(text):
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    # also some literal fallbacks just in case
    text = text.replace("The average material cost is 64% of the sales value. The only 36% cost is for wages and salaries, overhead and profit, etc.", "")
    text = text.replace("The average material cost is 64% of the sales Value. The only 36% cost is for wages and salaries, overhead and profit, etc.", "")
    # clean up empty markdown headers
    text = text.replace("## Key Metrics\n\n\n\n---\n\n", "")
    text = text.replace("## Key Metrics\n\n---\n\n", "")
    return text.strip()

for note_id, note in db.items():
    for field in ['raw_markdown', 'verified_markdown']:
        if field in note and note[field]:
            note[field] = clean_text(note[field])

with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2)

print("Database cleaned!")

# Clean text files in output dir
output_files = glob.glob('backend/output/*.txt')
for filepath in output_files:
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_content = clean_text(content)
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Cleaned {filepath}")
