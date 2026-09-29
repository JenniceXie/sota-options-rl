"""Build the static project page from the approved SVG and published CSV.

Run: python scripts/build_project_page.py
Only the Python standard library is required. No network requests are made.
"""
from pathlib import Path
import csv
import html
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
ET.register_namespace('', 'http://www.w3.org/2000/svg')
diagram = ET.parse(DOCS / 'sota_framework.svg').getroot()
diagram.set('role', 'img')
diagram.set('aria-labelledby', 'diagram-title diagram-description')
ns = '{http://www.w3.org/2000/svg}'
title = ET.Element(ns + 'title', {'id': 'diagram-title'})
title.text = 'SOTA: strategy selection with supervised and reinforcement learning'
description = ET.Element(ns + 'desc', {'id': 'diagram-description'})
description.text = ('Market observations feed the frontier teacher and strategy selector. '
                    'The teacher supervises the student. Both propose candidate strategies. '
                    'A selected strategy is implemented in a portfolio; reward updates the student.')
diagram.insert(0, title)
diagram.insert(1, description)
node_positions = {'375': 'observations', '70': 'teacher', '890': 'student',
                  '475': 'selection', '430': 'implementation'}
flow_starts = {'600,93': 'observations', '600,120': 'observations',
               '310,267': 'sft', '610,165': 'sft', '310,310': 'teacher',
               '890,310': 'student', '600,340': 'implementation', '770,466': 'feedback'}
for element in diagram:
    if element.tag == ns + 'rect' and element.get('x') in node_positions:
        element.set('data-node', node_positions[element.get('x')])
    if element.tag == ns + 'polyline':
        key = element.get('points', '').split(' ')[0]
        if key in flow_starts:
            element.set('data-flow', flow_starts[key])
diagram_markup = ET.tostring(diagram, encoding='unicode')

with (ROOT / 'results/table1.csv').open(encoding='utf-8', newline='') as stream:
    results = list(csv.DictReader(stream))[:6]
rows = []
for row in results:
    row_class = ' class="ours"' if row['row'] == 'SOTA' else ''
    cells = [html.escape(row['row'])] + [f'{float(row[key]):.2f}' for key in ['TR_pct', 'ASR', 'MDD_pct']]
    rows.append(f'<tr{row_class}>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
labels = ['Observe', 'Generate', 'Supervise', 'Select', 'Implement', 'Improve']
buttons = ''.join(f'<button class="step" type="button" aria-controls="step-text"><span>{i+1}</span>{label}</button>' for i, label in enumerate(labels))
page = '''<!doctype html>
<html lang="en" class="js-disabled"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOTA — Stock Options Trading Agents</title>
<meta name="description" content="Stock Options Trading Agents guided by option-implied return distributions. A two-stage learning framework for structured option strategy selection.">
<link rel="icon" href="sota_icon.svg" type="image/svg+xml">
<link rel="stylesheet" href="site.css"><script src="site.js" defer></script>
</head><body>
<a class="skip" href="#main">Skip to content</a>
<nav aria-label="Main navigation"><div class="nav-inner"><a class="brand" href="#main">SOTA</a><div class="nav-links"><a href="#framework">Framework</a><a href="#method">Method</a><a href="#results">Results</a><a href="https://github.com/JenniceXie/sota-options-rl">Code ↗</a></div></div></nav>
<main id="main"><header class="hero">
<h1 class="sr-only">SOTA: Stock Options Trading Agents Guided by Option-Implied Return Distributions</h1>
<img class="banner" src="sota_banner.svg" width="1440" height="420" alt="SOTA — Stock Options Trading Agents. Learn which option strategy to trade. Guided by option-implied return distributions: direction, volatility, skewness, and curvature.">
<p class="authors">Yizhen Xie · Mengyang Liu</p><div class="affiliations">Carnegie Mellon University · Amazon</div>
<div class="hero-actions"><a class="button primary" href="#framework">Explore the framework</a><a class="button" href="https://github.com/JenniceXie/sota-options-rl">GitHub repository ↗</a><a class="button" href="https://github.com/JenniceXie/sota-options-rl/blob/main/CITATION.cff">Citation</a></div>
<p class="intro">A price forecast is only the beginning. <strong>SOTA learns which option strategy to trade</strong>, selecting among nine structured strategy families. Deterministic resolvers handle contracts, sizing, and hedging.</p>
</header>
<section id="framework" aria-labelledby="framework-title"><div class="eyebrow">Framework</div><h2 id="framework-title">From market observations to a trading policy.</h2><p class="section-intro">The teacher provides a starting point. Portfolio rewards refine the student. Follow the six steps to see how the two training phases connect.</p>
<div class="panel" id="walkthrough"><div class="panel-top"><h3>How SOTA learns strategy selection</h3><div class="controls" role="group" aria-label="Animation controls"><button id="previous" type="button" aria-label="Previous step">←</button><button id="play" type="button" aria-pressed="false">Play</button><button id="next" type="button" aria-label="Next step">→</button></div></div>
<div class="steps" role="group" aria-label="Framework steps">{{BUTTONS}}</div>
<div class="diagram-wrap">{{DIAGRAM}}</div>
<div class="explanation" aria-live="polite" aria-atomic="true"><div><div class="step-label" id="step-count">Step 1 / 6</div><h3 id="step-title">Observe</h3></div><p id="step-text"></p></div>
<p class="no-js">This static diagram shows both training stages. The frontier teacher generates supervised trajectories; reward-driven reinforcement learning refines the student.</p>
<p class="figure-note">Simplified overview. News is used for teacher supervision; the reported RL policy uses market states only. The feedback arrow summarizes reward computation and policy optimization.</p>
</div><p class="downloads"><a href="sota_framework.svg" download>Download figure (SVG)</a><a href="sota_framework.png" download>Download figure (PNG)</a></p></section>
<section id="method" aria-labelledby="method-title"><div class="eyebrow">Method</div><h2 id="method-title">Two stages. One structured decision space.</h2><div class="method-grid"><article><h3>Phase I · Supervised fine-tuning</h3><p>A frontier LLM produces trading trajectories from anonymized market states and contemporaneous news. Trajectories pass an outcome gate before the student is trained on the teacher's decisions.</p></article><article><h3>Phase II · Reinforcement learning</h3><p>The student interacts with the same portfolio environment and learns from rewards net of transaction costs. The reported configuration uses market states only.</p></article></div>
<div class="cards"><article class="card"><h3>Structured strategies</h3><p>Nine option-strategy families express different directional, volatility, skewness, and curvature exposures.</p></article><article class="card"><h3>Deterministic execution</h3><p>Rule-based resolvers map decisions to contracts, position sizes, and delta hedges.</p></article><article class="card"><h3>Point-in-time inputs</h3><p>Features and news respect information-availability timestamps. Identity, price levels, and calendar time are anonymized.</p></article></div></section>
<section id="results" aria-labelledby="results-title"><div class="eyebrow">Reported results</div><h2 id="results-title">Six months out of sample.</h2><p class="section-intro">Options on SPY and nine large-cap U.S. equities. Test window: March 3–August 29, 2025. Values below come from the repository's results table.</p>
<div class="cards"><div class="card"><div class="stat">18.32%</div><div class="stat-label">Total return</div></div><div class="card"><div class="stat">1.60</div><div class="stat-label">Annualized Sharpe ratio</div></div><div class="card"><div class="stat">8.96%</div><div class="stat-label">Maximum drawdown</div></div></div>
<div class="table-wrap"><table><caption class="sr-only">Test-window comparison of SOTA and five baselines</caption><thead><tr><th scope="col">Policy</th><th scope="col">Total return (%)</th><th scope="col">Sharpe ratio</th><th scope="col">Max. drawdown (%)</th></tr></thead><tbody>{{ROWS}}</tbody></table></div>
<p class="figure-note">SOTA: Qwen3.8-27B after supervised fine-tuning and reinforcement learning, at RL step 20. <a href="https://github.com/JenniceXie/sota-options-rl/blob/main/results/table1.csv">Full metrics and provenance ↗</a></p>
</section>
<footer>Research code and data documentation: <a href="https://github.com/JenniceXie/sota-options-rl">JenniceXie/sota-options-rl</a>. Visual styling inspired by <a href="https://dreamyang-liu.github.io/STORM/">STORM</a>; figures and page implementation are original to SOTA.</footer>
</main></body></html>'''
page = page.replace('{{BUTTONS}}', buttons).replace('{{DIAGRAM}}', diagram_markup).replace('{{ROWS}}', ''.join(rows))
(DOCS / 'index.html').write_text(page, encoding='utf-8')
print('Built docs/index.html from docs/sota_framework.svg and results/table1.csv')
