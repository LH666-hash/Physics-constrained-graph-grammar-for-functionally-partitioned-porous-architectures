"""Export the six final embedded main figures as standalone PNGs."""
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

P=Path(__file__).resolve().parents[2]
doc=Document(P/"Manuscript/Manuscript.docx")
out=P/"Figures/Main_Figures"
locations={20:1,36:2,43:3,51:4,58:5,66:6}
for paragraph,number in locations.items():
    blips=list(doc.paragraphs[paragraph]._p.iter(qn("a:blip")))
    assert len(blips)==1,(paragraph,len(blips))
    rid=blips[0].get(qn("r:embed"))
    data=doc.part.related_parts[rid].blob
    (out/f"Figure{number}.png").write_bytes(data)
print("Exported final main Figures 1–6 from the revised Word file.")
