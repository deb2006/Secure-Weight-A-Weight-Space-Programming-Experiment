(() => {
  const {study} = JSON.parse(document.getElementById('data').textContent);
  const byId = id => document.getElementById(id);
  const trials = study.trials;
  const networkInfo = {
    iris: {name: 'Iris', inputs: 4, parameters: 67, task: 'Classify a flower as setosa, versicolor or virginica from four sepal and petal measurements.', split: [91, 22, 15, 22], featured: 12},
    wine: {name: 'Wine', inputs: 13, parameters: 139, task: 'Identify one of three wine cultivars from thirteen chemical measurements, including alcohol, flavonoids and color intensity.', split: [106, 27, 18, 27], featured: 6}
  };
  const build = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const idText = n => String(n).padStart(2, '0');
  const fmt = value => value.toFixed(4).replace(/^-/, '−');
  const keyOf = index => index.join(',');
  let trial = trials.find(t => t.trial === 6);
  let coordinate = [2, 1];
  let heroState = 'after';

  const damaged = trials.filter(t => t.condition === 'damaged');
  const healthy = trials.filter(t => t.condition === 'healthy');
  const metricData = [
    [damaged.filter(t => t.suspicion === 'YES').length, damaged.length, 'Damaged copies flagged'],
    [healthy.filter(t => t.suspicion === 'YES').length, healthy.length, 'Healthy copies flagged'],
    [damaged.filter(t => t.edits.length).length, damaged.length, 'Damaged copies edited'],
    [damaged.filter(t => t.test.after > t.test.given).length, damaged.length, 'Held-out scores improved']
  ];
  for (const [amount, total, label] of metricData) {
    const item = build('div', undefined, 'metric');
    const value = build('strong', amount);
    value.append(build('small', ' / ' + total));
    item.append(value, build('span', label));
    byId('metrics').append(item);
  }
  for (const info of Object.values(networkInfo)) {
    const article = build('article', undefined, 'network');
    const head = build('div', undefined, 'panel-head');
    head.append(build('h3', info.name), build('small', info.parameters + ' parameters'));
    const architecture = build('div', undefined, 'architecture');
    [[info.inputs, 'inputs'], [8, 'ReLU units'], [3, 'outputs']].forEach(([n, name], i) => {
      if (i) architecture.append(build('i', undefined, 'connector'));
      const layer = build('div', undefined, 'layer');
      layer.append(build('b', n), build('span', name));
      architecture.append(layer);
    });
    article.append(head, build('p', info.task), architecture,
      build('p', info.split.map((n, i) => n + ' ' + ['train', 'probe', 'validation', 'test'][i]).join(' · '), 'splitline'));
    byId('networkCards').append(article);
  }

  function tabulate(table, headings, records) {
    table.querySelectorAll('thead,tbody').forEach(group => group.remove());
    const tr = table.createTHead().insertRow();
    for (const heading of headings) {
      const th = build('th', heading);
      th.scope = 'col';
      tr.append(th);
    }
    const body = table.createTBody();
    for (const fields of records) {
      const row = body.insertRow();
      fields.forEach((value, i) => {
        const cell = build(i ? 'td' : 'th', value);
        if (!i) cell.scope = 'row';
        row.append(cell);
      });
    }
  }

  function drawPilots() {
    const split = byId('pilotSplit').value;
    const fields = study.pilots.map(p => {
      const series = p[split];
      return [p.dataset, p.fault, p.decision + ' ' + p.edits,
        ...['healthy', 'damaged', 'repaired'].map(s => series ? series[s] + '/' + series.n : '—')];
    });
    tabulate(byId('pilotTable'), ['Network', 'Fault', 'First reply', 'Healthy', 'Damaged', 'After'], fields);
    for (const p of study.pilots.filter(item => item.fault === 'Hidden layer erased')) {
      const score = p[split];
      const figure = byId(p.dataset.toLowerCase() + 'PilotChart');
      const caption = build('figcaption', (split === 'test' ? 'Held-out' : 'Validation') + ' accuracy');
      const bars = ['healthy', 'damaged', 'repaired'].map(state => {
        const line = build('div', undefined, 'pilot-bar ' + state);
        const track = build('span', undefined, 'track');
        const bar = build('i');
        bar.style.width = (100 * score[state] / score.n) + '%';
        track.setAttribute('aria-hidden', 'true');
        track.append(bar);
        line.append(build('span', state[0].toUpperCase() + state.slice(1)), track, build('b', score[state] + '/' + score.n));
        return line;
      });
      figure.replaceChildren(caption, ...bars);
    }
    byId('pilotNote').textContent = split === 'test'
      ? 'Held-out testing was recorded for the two hidden-layer repairs. A dash indicates an unmeasured split.'
      : 'Correct / total validation examples. Early blank replies were logged as missing.';
  }

  function renderTopology() {
    const info = networkInfo[byId('heroDataset').value];
    const t = trials.find(r => r.trial === info.featured);
    const canvas = byId('topology');
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    const cols = [info.inputs, 8, 3].map((count, layer) => {
      const x = [w * .13, w * .50, w * .87][layer];
      const span = (h - 50) * [1, .87, .55][layer];
      return Array.from({length: count}, (_, i) => [x, (h - 15) / 2 + (i / (count - 1) - .5) * span]);
    });
    const altered = new Map(t.changes.map(c => [keyOf(c.index), c]));
    const edits = new Map(t.edits.map(e => [keyOf(e.index), e]));
    function wire(a, b, weight, special) {
      ctx.beginPath();
      ctx.moveTo(...a);
      const dx = (b[0] - a[0]) * .44;
      ctx.bezierCurveTo(a[0]+dx,a[1],b[0]-dx,b[1],...b);
      ctx.strokeStyle = special === 'edit' ? '#9bfbd4' : special === 'fault' ? '#edaa79'
        : 'rgba(143,198,174,' + (.15 + Math.min(Math.abs(weight), 1.8)*.17) + ')';
      ctx.lineWidth = special ? 2.2 : .55 + Math.min(Math.abs(weight), 1.8)*.57;
      ctx.setLineDash(special === 'fault' ? [4, 3] : []);
      ctx.stroke();
    }
    cols[0].forEach((a, i) => cols[1].forEach((b,j) => wire(a,b,t.hidden[j][i])));
    const marked = [];
    cols[1].forEach((a,i) => cols[2].forEach((b,j) => {
      const k=keyOf([j,i]), fault=altered.get(k), edit=edits.get(k);
      const special = fault && heroState !== 'healthy' ? (edit && heroState === 'after' ? 'edit' : 'fault') : null;
      const weight = fault && heroState === 'healthy' ? fault.healthy : edit && heroState === 'after' ? edit.after : t.output[j][i];
      if (special) marked.push([a,b,weight,special]);
      else wire(a,b,weight);
    }));
    marked.forEach(args => wire(...args));
    ctx.setLineDash([]);
    cols.forEach((nodes, layer) => {
      const r = [3.4, 5, 6.8][layer];
      nodes.forEach(([x,y]) => {
        ctx.beginPath();ctx.arc(x,y,r+2.5,0,2*Math.PI);ctx.fillStyle='#0d1511';ctx.fill();
        ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);ctx.fillStyle=['#83bca4','#b6e6ca','#d7fae2'][layer];ctx.fill();
      });
      ctx.fillStyle='#c8d4cd';ctx.font='12px Segoe UI, sans-serif';ctx.textAlign='center';
      ctx.fillText([info.inputs+' inputs','8 hidden','3 outputs'][layer],nodes[0][0],h-3);
    });
    const action = heroState === 'healthy' ? 'Healthy' : heroState === 'damaged' ? 'Damaged' : 'After edits';
    byId('heroState').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.state === heroState));
    byId('wireLegend').textContent = {healthy:'Intact connections',damaged:'Dashed · injected changes',after:'Mint · edited   /   dashed · left damaged'}[heroState];
    byId('heroCaption').textContent = info.name + ' · ' + action + ' checkpoint. ' + t.test[heroState] + ' of ' + t.test.total + ' held-out examples classified correctly. Line widths on intact connections reflect weight magnitude.';
    canvas.setAttribute('aria-label', info.name+' network, '+action.toLowerCase()+' checkpoint. '+byId('wireLegend').textContent+'.');
  }

  function drawPicker() {
    const filter = byId('trialFilter').value;
    const eligible = trials.filter(t => filter === 'all' || t.dataset === filter);
    if (!eligible.includes(trial)) trial = eligible[0];
    byId('trialPicker').replaceChildren(...eligible.map(t => {
      const button = build('button', 'Trial ' + idText(t.trial), 'trial-btn');
      button.append(build('small', networkInfo[t.dataset].name));
      button.setAttribute('aria-pressed', t === trial);
      button.onclick = () => {
        trial=t; drawTrial();
        byId('trialPicker').querySelector('[aria-pressed=true]').focus({preventScroll:true});
      };
      return button;
    }));
  }

  function drawWeightMap() {
    const changed = new Map(trial.changes.map(c => [keyOf(c.index),c]));
    const edits = new Map(trial.edits.map(e => [keyOf(e.index),e]));
    const map = byId('weightMap');
    map.replaceChildren(build('span', ''));
    trial.output[0].forEach((_,j) => map.append(build('span',j)));
    for (const [i,row] of trial.output.entries()) {
      map.append(build('span',i));
      row.forEach((value,j) => {
        const key=keyOf([i,j]), change=changed.get(key), edit=edits.get(key);
        const kind = edit ? 'edited' : change ? 'fault' : 'healthy';
        const b=build('button',edit?'◆':change?'○':'•','mapcell '+kind);
        b.dataset.coordinate=key;
        b.setAttribute('aria-label','Output weight ['+key+'], '+(kind==='fault'?'damaged and left alone':kind));
        b.setAttribute('aria-pressed', key===keyOf(coordinate));
        b.onclick=() => {coordinate=[i,j]; map.querySelectorAll('button').forEach(c=>c.setAttribute('aria-pressed', c===b)); readCoordinate();};
        map.append(b);
      });
    }
    function readCoordinate() {
      const [i,j]=coordinate, k=keyOf(coordinate), given=trial.output[i][j];
      const base=changed.get(k)?.healthy ?? given, after=edits.get(k)?.after ?? given;
      byId('weightReadout').textContent='['+k+']  Healthy '+fmt(base)+' · Shown '+fmt(given)+' · After '+fmt(after);
      byId('weightReadout').title='Full precision: healthy '+base+', shown '+given+', after '+after;
    }
    readCoordinate();
  }

  function drawTrial() {
    drawPicker();
    const t=trial;
    coordinate=t.edits[0]?.index || t.changes[0]?.index || [0,0];
    const pair=trials.find(p=>p.dataset===t.dataset && p.training_seed===t.training_seed && p.condition!==t.condition);
    byId('trialHeading').replaceChildren(build('h3','Trial '+idText(t.trial)+' · '+networkInfo[t.dataset].name+' · '+t.condition),build('span','Seed '+t.training_seed+' / paired with '+idText(pair.trial)));
    const first=t.edits.length?'EDIT '+t.edits.length:'KEEP 0';
    byId('decisions').replaceChildren(...[['Original action',first],['Suspected damage?',t.suspicion],['Follow-up action',t.action]].map(([label,value])=>{
      const line=build('div');line.append(build('dt',label),build('dd',value));return line;
    }));
    byId('explanation').textContent=t.reason;
    const table=build('table');
    tabulate(table,['Split','Healthy','Shown','After'],['validation','test'].map(split=>{
      const score=t[split];
      return [split==='test'?'Held-out':'Validation',...['healthy','given','after'].map(k=>score[k]+'/'+score.total)];
    }));
    byId('trialAccuracy').replaceChildren(table,build('p','Shown = checkpoint sent to the LLM. After = first-reply edits applied.', 'note'));
    drawWeightMap();
  }
  byId('heroDataset').onchange=renderTopology;
  byId('heroState').querySelectorAll('button').forEach(b=>b.onclick=()=>{heroState=b.dataset.state;renderTopology();});
  new ResizeObserver(renderTopology).observe(byId('topology'));
  byId('pilotSplit').onchange=drawPilots;
  byId('trialFilter').onchange=drawTrial;
  drawPilots();drawTrial();renderTopology();
})();
