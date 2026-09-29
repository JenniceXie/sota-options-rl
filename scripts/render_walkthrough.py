"""Render the vector walkthrough at 3000px wide, without screenshot scaling.

Requires Pillow. Run after build_project_page.py. Uses Segoe UI on Windows;
set SOTA_FONT_DIR to a folder containing segoeui.ttf and segoeuib.ttf elsewhere.
The small renderer supports the SVG primitives used by this project's figure.
"""
from pathlib import Path
import copy
import os
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
SCALE = 2.5
WIDTH, HEIGHT = 1200, 900
FONT_DIR = Path(os.environ.get('SOTA_FONT_DIR', 'C:/Windows/Fonts'))
page = (DOCS / 'index.html').read_text(encoding='utf-8')
svg = ET.fromstring(re.search(r'<svg\b.*?</svg>', page, re.S).group())
labels = ['Observe', 'Generate', 'Supervise', 'Select', 'Implement', 'Improve']
nodes = [['observations'], ['teacher', 'selection'], ['teacher', 'student'],
         ['student', 'selection'], ['selection', 'implementation'], ['implementation', 'student']]
flows = ['observations', 'teacher', 'sft', 'student', 'implementation', 'feedback']

def numbers(s):
    return [float(n) for n in re.findall(r'-?\d*\.?\d+', s)]

def render(stage):
    root = copy.deepcopy(svg)
    expanded = stage == 3
    for el in root.iter():
        node = el.get('data-node')
        if node == 'selection':
            el.attrib.update(fill='#d9f1e5', stroke='#0f7a55', **{'stroke-width':'3.5'})
        if node in nodes[stage]:
            fill, stroke = {'teacher':('#e9f2ff','#2a78d6'), 'student':('#fff0e8','#eb6834'),
                            'selection':('#c7ebd9','#0f7a55')}.get(node, ('#d9f1e5','#0f7a55'))
            el.attrib.update(fill=fill, stroke=stroke, **{'stroke-width':'5' if node == 'selection' else '4'})
        flow = el.get('data-flow')
        if flow == flows[stage]:
            el.attrib.update(stroke='#0f7a55', **{'stroke-width':'3'})
        if expanded:
            if el.get('data-lower'):
                el.set('transform', 'translate(0 220)')
            if flow == 'implementation':
                el.set('points', '600,580 600,640')
            if flow == 'feedback':
                el.set('points', '770,686 1192,686 1192,350 1010,350 1010,340')
    image = Image.new('RGB', (int(WIDTH*SCALE), int(HEIGHT*SCALE)), 'white')
    draw = ImageDraw.Draw(image)

    def text(x,y,value,size=18,bold=False,color='#0e1512',anchor='ls'):
        font=ImageFont.truetype(str(FONT_DIR / ('segoeuib.ttf' if bold else 'segoeui.ttf')), round(size*SCALE))
        draw.text((x*SCALE,y*SCALE),value,font=font,fill=color,anchor=anchor)

    def element(el, dx=0, dy=105):
        tag = el.tag.rsplit('}',1)[-1]
        if tag in ('title','desc') or (el.get('id') == 'strategy-expansion' and not expanded):
            return
        if el.get('transform'):
            tx,ty=numbers(el.get('transform')); dx+=tx;dy+=ty
        def point(x,y): return ((x+dx)*SCALE,(y+dy)*SCALE)
        fill=el.get('fill','black'); stroke=el.get('stroke')
        if fill=='none': fill=None
        if stroke=='none': stroke=None
        width=max(1,round(float(el.get('stroke-width','1'))*SCALE))
        if tag=='rect':
            x=float(el.get('x',0));y=float(el.get('y',0));w=float(el.get('width'));h=float(el.get('height'))
            draw.rounded_rectangle([point(x,y),point(x+w,y+h)],radius=float(el.get('rx',0))*SCALE,fill=fill,outline=stroke,width=width)
        elif tag=='text':
            text(float(el.get('x',0))+dx,float(el.get('y',0))+dy,el.text or '',float(el.get('font-size','16')),int(el.get('font-weight','400'))>=600,fill,{'middle':'ms','end':'rs'}.get(el.get('text-anchor'),'ls'))
        elif tag in ('polyline','polygon'):
            coords=numbers(el.get('points')); pts=[point(*coords[i:i+2]) for i in range(0,len(coords),2)]
            if tag=='polygon': draw.polygon(pts,fill=fill)
            if stroke: draw.line(pts+([pts[0]] if tag=='polygon' else []),fill=stroke,width=width,joint='curve')
        elif tag=='path':
            tokens=re.findall(r'[MHVC]|-?\d*\.?\d+',el.get('d'))
            i=0;x=y=0;pts=[]
            while i<len(tokens):
                command=tokens[i];i+=1
                count={'M':2,'H':1,'V':1,'C':6}[command]
                vals=list(map(float,tokens[i:i+count]));i+=count
                if command=='M':
                    if len(pts)>1: draw.line(pts,fill=stroke,width=width,joint='curve')
                    x,y=vals;pts=[point(x,y)]
                elif command=='H': x=vals[0];pts.append(point(x,y))
                elif command=='V': y=vals[0];pts.append(point(x,y))
                else:
                    x1,y1,x2,y2,x3,y3=vals
                    for j in range(1,33):
                        t=j/32;u=1-t
                        pts.append(point(u**3*x+3*u*u*t*x1+3*u*t*t*x2+t**3*x3,u**3*y+3*u*u*t*y1+3*u*t*t*y2+t**3*y3))
                    x,y=x3,y3
            if len(pts)>1: draw.line(pts,fill=stroke,width=width,joint='curve')
        for child in el: element(child,dx,dy)
    element(root)
    text(20,32,'How SOTA learns strategy selection',22,True)
    for i,label in enumerate(labels):
        x=20+i*154;active=i==stage
        draw.rounded_rectangle((x*SCALE,52*SCALE,(x+144)*SCALE,90*SCALE),radius=19*SCALE,fill='#e7f5ee' if active else 'white',outline='#0f7a55' if active else '#e3e8e5',width=round(1.5*SCALE))
        text(x+72,77,f'{i+1}  {label}',16,active,'#0f7a55' if active else '#65716b','ms')
    return image

frames=[render(i) for i in range(6)]
frames[0].save(DOCS/'sota_walkthrough.gif',save_all=True,append_images=frames[1:],duration=1800,loop=0,optimize=True,disposal=2)
frames[3].save(DOCS/'sota_strategy_expanded.png',optimize=True)
print(f'Rendered {len(frames)} frames at {frames[0].size}, 1800 ms per stage')
