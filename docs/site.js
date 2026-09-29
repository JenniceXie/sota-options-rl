(() => {
  'use strict';
  const steps = [
    {title:'Observe', nodes:['observations'], flows:['observations'], text:'Build point-in-time market states from option-implied return distributions and other market features. The frontier teacher also receives contemporaneous news. Stock identity, absolute price levels, and calendar time are anonymized.'},
    {title:'Generate', nodes:['teacher','selection'], flows:['teacher'], text:'The frontier LLM proposes structured option-strategy decisions and runs them through the trading environment. Complete trajectories must pass a risk-adjusted-return and drawdown gate before they enter the supervised corpus.'},
    {title:'Supervise', nodes:['teacher','student'], flows:['sft'], text:'Phase I: supervised fine-tuning initializes the student from the accepted frontier-teacher trajectories. The training loss is applied to assistant responses, so the model learns trading decisions.'},
    {title:'Select', nodes:['student','selection'], flows:['student'], text:'The student selects among nine option-strategy families and specifies the underlying, tenor, and delta coordinates. In the reported Phase II configuration, the student uses market states only; news is not included.'},
    {title:'Implement', nodes:['selection','implementation'], flows:['implementation'], text:'Deterministic resolvers turn the selected strategy into contracts, position sizes, and delta hedges. The same portfolio environment is used for training and evaluation.'},
    {title:'Improve', nodes:['implementation','student'], flows:['feedback'], text:'Phase II: reinforcement learning uses portfolio rewards net of transaction costs to update the student policy. The diagram shows the causal sequence; it does not replay real trades or training measurements.'}
  ];
  const root = document.querySelector('#walkthrough');
  if (!root) return;
  const buttons = [...root.querySelectorAll('.step')];
  const play = root.querySelector('#play');
  const previous = root.querySelector('#previous');
  const next = root.querySelector('#next');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let current = 0;
  let timer = null;
  let playing = false;
  function render() {
    root.dataset.playing = String(playing);
    const step = steps[current];
    buttons.forEach((button, i) => {
      if (i === current) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    root.querySelectorAll('[data-node]').forEach(el => el.classList.toggle('active', step.nodes.includes(el.dataset.node)));
    root.querySelectorAll('[data-flow]').forEach(el => el.classList.toggle('active', step.flows.includes(el.dataset.flow)));
    root.querySelector('#step-count').textContent = `Step ${current + 1} / ${steps.length}`;
    root.querySelector('#step-title').textContent = step.title;
    root.querySelector('#step-text').textContent = step.text;
    previous.disabled = current === 0;
    next.disabled = current === steps.length - 1;
  }
  function pause() {
    playing = false;
    window.clearTimeout(timer);
    timer = null;
    play.textContent = 'Play';
    play.setAttribute('aria-pressed', 'false');
    root.dataset.playing = 'false';
  }
  function schedule() {
    timer = window.setTimeout(() => {
      if (!playing) return;
      if (current === steps.length - 1) { pause(); return; }
      current += 1;
      render();
      schedule();
    }, 5500);
  }
  play.addEventListener('click', () => {
    if (playing) { pause(); return; }
    if (current === steps.length - 1) current = 0;
    playing = true;
    play.textContent = 'Pause';
    play.setAttribute('aria-pressed', 'true');
    render();
    schedule();
  });
  previous.addEventListener('click', () => { pause(); current = Math.max(0, current - 1); render(); });
  next.addEventListener('click', () => { pause(); current = Math.min(steps.length - 1, current + 1); render(); });
  buttons.forEach((button, i) => button.addEventListener('click', () => { pause(); current = i; render(); }));
  root.addEventListener('keydown', event => {
    if (event.target.tagName !== 'BUTTON') return;
    if (!['ArrowLeft','ArrowRight'].includes(event.key)) return;
    event.preventDefault(); pause();
    current = Math.max(0, Math.min(steps.length - 1, current + (event.key === 'ArrowRight' ? 1 : -1)));
    render(); buttons[current].focus();
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) pause(); });
  // Start paused so the diagram is readable and never moves unexpectedly.
  render();
  document.documentElement.classList.remove('js-disabled');
})();
