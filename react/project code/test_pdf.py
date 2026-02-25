import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'backend'))

from services.export_service import export_pdf, export_docx

md_text = """# Web Tech Test
    
This is a **bold text** and this is regular.

## Features
- First item
- Second item

| Name | Role | Location |
|---|---|---|
| Alice | Admin | NY |
| Bob | User | CA |

1. Numbered one
2. Numbered two
"""

try:
    pdf_bytes = export_pdf(md_text, "Formating Table PDF Test")
    print(f"Success! Generated {len(pdf_bytes)} bytes PDF")
    
    docx_bytes = export_docx(md_text, "Formatting Table DOCX Test")
    print(f"Success! Generated {len(docx_bytes)} bytes DOCX")
except Exception as e:
    import traceback
    traceback.print_exc()
