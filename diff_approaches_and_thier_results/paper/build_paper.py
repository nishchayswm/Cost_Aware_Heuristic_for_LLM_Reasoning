"""Build an editable research manuscript from audited evidence, without model calls."""
from pathlib import Path
import json,re
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from reportlab.graphics.shapes import Drawing,String,Rect,Line
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics import renderSVG
from reportlab.lib import colors

ROOT=Path(__file__).resolve().parent
e=json.loads((ROOT/'evidence.json').read_text())
original=e['game24_original_n100_seed42_ollama']['summary'];generated=e['generated_summary']
names={'heuristic_bestfirst':'Five factor heuristic','quality_bestfirst':'Quality only best first',
       'tot_bfs':'Bounded breadth search','self_eval_beam':'Stochastic beam',
       'cot':'Single expression CoT','self_consistency':'Fixed verified sampling',
       'verified_sampling':'Early verified sampling','symbolic_search':'Exact symbolic search'}
order=list(names)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])
tables={}
tables['METHOD_TABLE']='Table 1 Operational comparator definitions\n\n'+table(['Method','Selection or stopping rule','Shared machinery'],[
 ['Five factor heuristic','Highest composite priority; active width 12','Legal children, ratings, rollouts, endgame'],
 ['Quality only best first','Highest Q; active width 12','Same as heuristic'],
 ['Bounded breadth search','Minimum active depth then highest P; width 2','Same engine, different frontier rule'],
 ['Stochastic beam','Minimum active depth, sample exp(P/0.3); width 2','Same engine, different selection'],
 ['Single expression CoT','One sampled reasoning and expression response','Exact final verification only'],
 ['Fixed verified sampling','Six full responses; accept any verified solution','Exact final verification only'],
 ['Early verified sampling','At most six responses; stop at first success','Exact final verification only'],
 ['Exact symbolic search','Depth first enumeration until success or exhaustion','Exact arithmetic and local endgame; no model']])
tables['ORIGINAL_TABLE']='Table 2 Original ToT split with 100 puzzles and seed 42\n\n'+table(
 ['Method','Solved','Calls','Tokens','Time s'],
 [[names[m],f'{original[m]["solved"]}/100',f'{original[m]["calls"]:.2f}',f'{original[m]["total_tokens"]:,.2f}',
   f'{original[m]["execution_seconds"]:.4f}' if m=='symbolic_search' else f'{original[m]["execution_seconds"]:.2f}'] for m in order])
tables['GENERATED_TABLE']='Table 4 Generated test set with 100 puzzles and three seeds\n\n'+table(
 ['Method','Seed successes','Mean success','Calls','Tokens'],
 [[names[m],'/'.join(map(str,generated[m]['seed_successes'])),f'{100*generated[m]["correct"]:.1f}%',
   f'{generated[m]["calls"]:.2f}',f'{generated[m]["total_tokens"]:,.2f}'] for m in order])
tables['DISJOINT_TABLE']='Table 5 Supplementary original subset without generated split overlap\n\n'+table(
 ['Method','Solved of 81','Success','Calls','Tokens'],
 [[names[m],str(e['disjoint_summary'][m]['solved']),f'{100*e["disjoint_summary"][m]["solved"]/81:.1f}%',
   f'{e["disjoint_summary"][m]["calls"]:.2f}',f'{e["disjoint_summary"][m]["tokens"]:,.2f}'] for m in order])
rows=[]
for m in ('quality_bestfirst','tot_bfs','self_eval_beam'):
    p=e['paired_original'][m]
    rows.append([names[m],f'{p["delta"]*100:+.1f}',f'{p["delta_ci"][0]*100:+.1f} to {p["delta_ci"][1]*100:+.1f}',
      f'{p["token_saving"]*100:.1f}%',f'{p["token_saving_ci"][0]*100:.1f} to {p["token_saving_ci"][1]*100:.1f}'])
tables['PAIRED_TABLE']='Table 3 Paired original split differences and 95 percent bootstrap intervals\n\n'+table(
 ['Baseline','Accuracy difference pp','Difference interval pp','Token saving','Saving interval percent'],rows)
run_names=[n for n,v in e.items() if isinstance(v,dict) and 'manifest' in v]
tables['RUN_TABLE']='Table A1 Included directories beneath final_proj results\n\n'+table(
 ['Directory','Puzzles','Seed','Records'],[[n,'100',e[n]['manifest']['config']['seed'],'800'] for n in run_names])
tables['CONFIG_TABLE']='Table B1 Effective Game of 24 settings\n\n'+table(['Setting','Value'],[
 ['Model tag','qwen2.5:7b-instruct'],['Model calls per method and puzzle','32 maximum'],['Total input plus output tokens','30,000 maximum'],
 ['Total generated tokens','6,144 maximum'],['Expansion limit','64'],['Context setting','8,192'],['Best first active frontier','12'],
 ['Breadth and stochastic active width','2'],['Fixed and early sampling cap','6 expressions'],['Rating shortlist','At most 6 children'],
 ['Rating output limit','700 tokens'],['Rollout output limit','260 tokens'],['Full expression output limit','1,000 tokens'],
 ['Rollout allowance','2 calls total per method and puzzle'],['Cost reference','4,000 tokens'],
 ['Quality success depth cost diversity weights','0.35 / 0.30 / 0.05 / 0.15 / 0.15'],
 ['Sampling rating rollout temperatures','0.7 / 0 / 0.4'],['Stochastic beam temperature','0.3']])

# Standard ReportLab charts, exported as vector SVG and rasterized externally for Word.
d=Drawing(640,270)
for panel,(key,title,maxval) in enumerate([('correct','Verified success percent',100),('total_tokens','Mean input plus output tokens',9000)]):
    x=150+panel*320
    d.add(String(x,250,title,fontName='Helvetica-Bold',fontSize=10))
    d.add(Line(x,30,x+150,30,strokeColor=colors.black))
    for index,m in enumerate(order):
        y=220-index*25
        val=original[m][key]*100 if key=='correct' else original[m][key]
        d.add(String(panel*320+5,y+5,names[m],fontName='Helvetica',fontSize=8))
        d.add(Rect(x,y,150*val/maxval,16,fillColor=colors.HexColor('#375774'),strokeColor=None))
    step=20 if panel==0 else 3000
    for tick in range(0,maxval+1,step):
        tx=x+150*tick/maxval
        d.add(Line(tx,27,tx,30,strokeColor=colors.black))
        d.add(String(tx,17,str(tick),textAnchor='middle',fontName='Helvetica',fontSize=8))

renderSVG.drawToFile(d,str(ROOT/'original_comparison.svg'))
tables['FIGURE_ORIGINAL']='![Figure 1 Verified success and total token cost on the original split](original_comparison.png)\n\nFigure 1 Success and cost are separate outcomes. The symbolic reference reaches 100 percent success with zero model tokens; its CPU work is not zero.'
text=(ROOT/'manuscript_template.md').read_text(encoding='utf-8')
for key,value in tables.items():text=text.replace('{{'+key+'}}',value)
assert '{{' not in text
(ROOT/'manuscript.md').write_text(text,encoding='utf-8')

def build_docx():
    doc=Document();sec=doc.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=Inches(.75);sec.left_margin=sec.right_margin=Inches(.8)
    for name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption'):
        st=doc.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
        fonts=st.element.get_or_add_rPr().find(qn('w:rFonts'))
        if fonts is not None:
            for attr in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):
                fonts.attrib.pop(qn('w:'+attr),None)
    # The stock template carries a blue Title border; remove inherited decoration.
    for element in doc.styles.element.iter(qn('w:pBdr')):
        element.getparent().remove(element)
    norm=doc.styles['Normal'];norm.font.size=Pt(11)
    norm.paragraph_format.line_spacing=1.08;norm.paragraph_format.space_after=Pt(6)
    norm.paragraph_format.widow_control=True
    for name,size in [('Title',20),('Heading 1',14),('Heading 2',12)]:
        st=doc.styles[name];st.font.size=Pt(size);st.font.bold=True
        st.paragraph_format.keep_with_next=True
        st.paragraph_format.space_before=Pt(12 if name!='Title' else 0)
        st.paragraph_format.space_after=Pt(6)
    doc.styles['Caption'].font.size=Pt(9.5)
    header=sec.header.paragraphs[0];header.text='Cost Aware Heuristic Selection for Arithmetic Search'
    header.style='Caption';header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('Research manuscript draft  |  ')
    fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
    for r in footer.runs:r.font.size=Pt(9)
    lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            i+=1;code=[]
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            for c in code:
                p=doc.add_paragraph();p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1
                r=p.add_run(c);r.font.name='Consolas';r.font.size=Pt(8)
            i+=1;continue
        if line.startswith('|'):
            tablelines=[]
            while i<len(lines) and lines[i].startswith('|'):tablelines.append(lines[i]);i+=1
            records=[[s.strip() for s in x.strip('|').split('|')] for j,x in enumerate(tablelines) if j!=1]
            t=doc.add_table(rows=0,cols=len(records[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
            n=len(records[0]);widths=([2.5,4.4] if n==2 else [3.9,1.,1.,1.] if n==4 and 'Directory' in records[0] else
                [1.65,2.6,2.65] if n==3 else [2.25]+[4.65/(n-1)]*(n-1))
            for col,w in zip(t.columns,widths):col.width=Inches(w)
            for j,row in enumerate(records):
                cells=t.add_row().cells
                for k,(cell,value) in enumerate(zip(cells,row)):
                    cell.width=Inches(widths[k]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    cell.text=value;p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(4)
                    p.paragraph_format.line_spacing=1.0
                    if k and n>=4:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:r.font.size=Pt(9);r.font.bold=j==0
                    props=cell._tc.get_or_add_tcPr()
                    borders=OxmlElement('w:tcBorders')
                    for side in ('top','left','bottom','right'):
                        b=OxmlElement('w:'+side);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
                    props.append(borders)
                    margins=OxmlElement('w:tcMar')
                    for side in ('top','left','bottom','right'):
                        m=OxmlElement('w:'+side);m.set(qn('w:w'),'80');m.set(qn('w:type'),'dxa');margins.append(m)
                    props.append(margins)
                    if j==0:
                        sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'DCE5ED');props.append(sh)
                trpr=t.rows[-1]._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
                if j==0:trpr.append(OxmlElement('w:tblHeader'))
            doc.add_paragraph().paragraph_format.space_after=Pt(1)
            continue
        if line.startswith('!['):
            path=ROOT/re.search(r'\]\(([^)]+)\)',line).group(1)
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            run=p.add_run();run.add_picture(str(path),width=Inches(6.85))
            for node in run._r.iter(qn('wp:docPr')):node.set('descr','Two horizontal bar charts showing verified success and mean total tokens for eight methods on the original 100-puzzle split.')
        elif line.startswith('$$'):
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            # Native Word math container: editable mathematics, no raw TeX in the document.
            mathp=OxmlElement('m:oMathPara');om=OxmlElement('m:oMath');mr=OxmlElement('m:r');mt=OxmlElement('m:t')
            mt.text=line.strip('$ ').strip();mr.append(mt);om.append(mr);mathp.append(om);p._p.append(mathp)
        elif line.startswith('# '):doc.add_paragraph(line[2:],'Title')
        elif line.startswith('### '):doc.add_heading(line[4:],2)
        elif line.startswith('## '):doc.add_heading(line[3:],1)
        else:
            p=doc.add_paragraph(line,'Caption' if line.startswith(('Table ','Figure ')) else None)
            if line.startswith('Table '):p.paragraph_format.keep_with_next=True
        i+=1
    doc.core_properties.title='Cost Aware Heuristic Selection for Language Model Guided Arithmetic Search'
    doc.core_properties.subject='Audited Game of 24 experiments and five-factor heuristic search'
    doc.core_properties.author=''
    doc.save(ROOT/'Cost_Aware_Heuristic_Research_Paper.docx')
    print('Created manuscript DOCX and Markdown. Words:',len(text.split()))

if __name__=='__main__':
    import sys
    if '--prepare-only' not in sys.argv:build_docx()

