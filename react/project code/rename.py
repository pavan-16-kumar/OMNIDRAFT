import os

base_dir = '/Users/pavankumar/Desktop/diploma/react/OmniDraft'
exclude_dirs = {'.git', 'venv', 'node_modules', 'output', 'data', '__pycache__', '.pytest_cache', 'build', 'dist'}

for root, dirs, files in os.walk(base_dir):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file in files:
        if file.endswith(('.py', '.js', '.jsx', '.html', '.md', '.json', '.css')):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                if 'OmniDraft' in content or 'omnidraft' in content or 'OMNIDRAFT' in content:
                    new_content = content.replace('OmniDraft', 'OmniDraft')
                    new_content = new_content.replace('omnidraft', 'omnidraft')
                    new_content = new_content.replace('OMNIDRAFT', 'OMNIDRAFT')

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated {filepath}")
            except Exception as e:
                pass
