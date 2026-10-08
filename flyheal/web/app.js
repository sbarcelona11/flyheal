import {frameIndex, latestLoader} from './playback.mjs';
import {BrainView, moduleStats, worldTransform} from './brain.mjs';
import {decodeDecision, topContributions} from './decision.mjs';
'use strict';
const $ = id => document.getElementById(id);
let selectedUnit=null,atlasPromise=null,orbitLast=0;
const brainView=new BrainView($('brain'),node=>{selectedUnit=node;brainView.module=node.module;$('unitSelect').value=String(ui.replay.population==='single'?node.index:node.layer*ui.replay.alive.length+node.index);render();});
const controls = ['architecture', 'seed', 'condition'];
const ui = {data: null, replay: null, stage: 'baseline', frame: 0, playing: false, edges: true, request: 0, last: 0};
const fmt = (n, digits=1) => Number(n).toLocaleString('es-UY', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const integer = n => Number(n).toLocaleString('es-UY');
const stageNames = {baseline: 'PRELESIÓN', post: 'LESIÓN PERMANENTE', recovered: 'TRAS REENTRENAR'};
const selection = () => ({architecture: $('architecture').value, seed: Number($('seed').value), condition: $('condition').value, stage: ui.stage});

async function json(url) {
  const response = await fetch(url);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'No se pudieron cargar los datos');
  return value;
}
function setOptions(element, options, selected) {
  element.replaceChildren(...options.map(([value, label]) => {
    const option = document.createElement('option'); option.value = value; option.textContent = label; return option;
  }));
  if (options.some(x => String(x[0]) === String(selected))) element.value = selected;
}
function updateMetrics() {
  if (!ui.data) return;
  const current = selection();
  const row = ui.data.seeds.find(r => r.architecture === current.architecture && r.seed === current.seed && r.condition === current.condition);
  if (!row) throw new Error('No hay métricas para esta selección');
  for (const key of ['pre', 'post', 'recovered', 'healthy']) $(key+'Metric').textContent = fmt(row[key]);
  $('dropMetric').textContent = row.drop > 0 ? `Caída de ${fmt(row.drop)} puntos` : row.drop < 0 ? `Mejora inmediata de ${fmt(-row.drop)} puntos` : 'Sin caída inmediata';
  $('gainMetric').textContent = `${row.gain >= 0 ? 'Ganancia' : 'Deterioro'} de ${fmt(Math.abs(row.gain))} puntos`;
  $('recoveryFraction').textContent = row.recovered_fraction === null ? 'No aplica' : fmt(row.recovered_fraction*100) + '%';
  $('chartSeed').textContent = current.seed;
  const foraging=ui.data.config.environment==='foraging';
  $('taskPanel').hidden=!foraging;
  if(foraging){const end=ui.data.config.recovery_steps;const rows=[['Antes','baseline',ui.data.config.baseline_steps],['Daño',current.condition,0],['Reentrenado',current.condition,end],['Control sano','healthy',end]];
    $('taskRows').replaceChildren(...rows.map(([label,condition,steps])=>{const row=ui.data.task_metrics.find(r=>r.architecture===current.architecture&&r.seed===current.seed&&r.condition===condition&&r.steps===steps);const tr=document.createElement('tr');for(const value of [label,row?fmt(row.mean_foods,2)+' / 3':'—',row?fmt(row.mean_collisions,1):'—',row?fmt(row.success_rate*100,0)+'%':'—']){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;}));
  }
  drawChart();
}
async function loadReplay() {
  const request = ++ui.request;
  ui.playing = false; ui.frame = 0; ui.replay = null;
  $('timeline').disabled = true; $('unitSelect').disabled=true; brainView.graph=null;brainView.frame=null;brainView.draw();
  $('play').disabled = true; $('play').textContent = '▶';
  $('play').setAttribute('aria-label', 'Reproducir episodio');
  $('replayStatus').textContent = 'Cargando checkpoint…';
  $('stageLabel').textContent = stageNames[ui.stage];
  document.querySelector('.brain-scene').classList.toggle('damaged', ui.stage === 'post');
  document.querySelectorAll('[data-stage]').forEach(button => {
    const selected = button.dataset.stage === ui.stage;
    button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', selected);
  });
  try {
    updateMetrics();
    const parameters = new URLSearchParams({...selection(), run: $('runSelect').value});
    const replay = await json('/api/replay?' + parameters);
    if (request !== ui.request) return;
    ui.replay = replay; ui.frame = 0; selectedUnit=null; brainView.setReplay(replay);
    if(replay.population==='single'){atlasPromise=atlasPromise||json('/api/atlas');brainView.setAtlas(await atlasPromise);if(request!==ui.request)return;}
    $('focusBrain').hidden=replay.population!=='single';
    setOptions($('unitSelect'),[['','Elegí una unidad'],...brainView.graph.nodes.map((node,index)=>[index,node.cell?`${node.cell.type} · ID ${node.cell.id}`:`Capa ${node.layer+1} · unidad ${node.index} · módulo ${'ABCD'[node.module]}`])],'');
    $('timeline').max = replay.frames.length - 1; $('timeline').value = 0;
    $('aliveCount').textContent = `${replay.alive.reduce((a,b)=>a+b,0)} / ${replay.alive.length} unidades activas`;
    $('connectionCount').textContent = integer(replay.active_connections);
    const real=replay.population==='single', measured=replay.provenance==='malecns-measured-subgraph';
    $('activityLegend').textContent=real?'Neuronas: verde + / naranja − · brillo = magnitud':'Color = módulo sintético · brillo = magnitud';
    $('layerOne').textContent=real?'80 neuronas · un circuito':'Capa 1 · '+replay.alive.length;
    $('layerTwo').textContent=real?'8 señales artificiales':'Capa 2 · '+replay.alive.length;
    document.querySelector('.provenance').textContent=measured?'Subgrafo MaleCNS validado · 80 neuronas':real?'Control recableado de MaleCNS':'Grafo sintético inspirado en conectomas';
    document.querySelector('.brain-disclaimer').textContent=measured?'COORDENADAS SOMÁTICAS Y CONECTIVIDAD MEDIDAS · ACTIVIDAD SIMULADA':real?'COORDENADAS MALECNS · CONEXIONES RECABLEADAS · ACTIVIDAD SIMULADA':'FORMA Y POSICIONES ILUSTRATIVAS · MÓDULOS SINTÉTICOS';
    document.querySelector('.network-head .eyebrow').textContent=real?'ACTIVIDAD POR GRUPO DE CÉLULAS':'ACTIVIDAD POR ZONA SINTÉTICA';
    document.querySelector('.network-head h3').textContent=real?'Explorá el circuito.':'Explorá cada módulo.';
    document.querySelector('.network-stat span').textContent=real?(measured?'Conexiones anatómicas retenidas':'Conexiones recableadas retenidas'):'Conexiones supervivientes · capa 2';
    document.querySelector('.network-note').textContent=real?`Media de |actividad|, no actividad biológica medida. ${integer(replay.modeled_connections)} conexiones con peso modelado no nulo. Baja actividad (<0,1) no es lesión. Hasta 650 aristas visibles; el contador usa todas.`:'Media de |tanh| por módulo, entre ambas capas. Baja actividad (<0,1) no es lesión. Hasta 650 aristas visibles; el contador usa todas.';
    document.querySelector('footer span:nth-child(2)').textContent=measured?'Subgrafo MaleCNS real · Dinámica y aprendizaje artificiales':real?'Control recableado · Actividad del modelo':'Grafo sintético · Actividad del modelo';
    $('brain').setAttribute('aria-label',real?'Subgrafo de 80 neuronas en sus coordenadas somáticas MaleCNS. Brillo muestra actividad simulada, cruces lesiones permanentes. Arrastrá para girar.':'Cerebro esquemático con cuatro módulos sintéticos. Actividad del modelo. Arrastrá para girar.');
    document.querySelectorAll('[data-module]').forEach((button,i)=>{button.hidden=real&&i===3;button.querySelector('b').textContent=real?['Visuales seleccionadas','Interneuronas','Salidas seleccionadas'][i]||'':`Módulo ${'ABCD'[i]}`;});
    $('returnValue').textContent = fmt(replay.return,replay.environment==='foraging'?1:0);
    $('timeline').max = replay.frames.length - 1;
    $('timeline').disabled = false; $('unitSelect').disabled=false;
    $('play').disabled = false;
    render();
  } catch (error) {
    if (request === ui.request) $('replayStatus').textContent = error.message;
  }
}
function canvasContext(id) {
  const canvas = $(id), box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  const width = box.width, height = box.height;
  if (canvas.width !== Math.round(width*ratio) || canvas.height !== Math.round(height*ratio)) {
    canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
  return {ctx, width, height};
}
function drawBrain(frame){
  brainView.edges=ui.edges;brainView.draw(frame);
  $('meanBrainActivity').textContent='Media |actividad| '+fmt(frame.h2.reduce((sum,v)=>sum+Math.abs(v),0)/frame.h2.length,3);
  $('atlasLegend').hidden=ui.replay.population!=='single';
  const stats=moduleStats(frame,ui.replay.alive,ui.replay.cells);
  stats.forEach((stat,i)=>{
    $('moduleValue'+i).textContent=fmt(stat.meanAbs,2);
    $('moduleBar'+i).value=stat.meanAbs;
    $('moduleCounts'+i).textContent=`${stat.silenced}/${stat.total} silenciadas · ${stat.low} de baja actividad`;
    $('moduleState'+i).textContent=stat.silenced?'Con lesión permanente':'Sin unidades silenciadas';
    const button=document.querySelector(`[data-module="${i}"]`);button.setAttribute('aria-pressed',brainView.module===i);
  });
  $('allModules').setAttribute('aria-pressed',brainView.module===null);
  if(selectedUnit){const n=selectedUnit,value=(n.layer?frame.h2:frame.h1)[n.index];$('unitDetail').textContent=(n.cell?`${n.cell.type} · ID ${n.cell.id} · NT ${n.cell.nt} · soma (voxeles 8 nm) [${n.cell.position.join(', ')}] · actividad ${fmt(value,3)}`:`Módulo ${'ABCD'[n.module]} · capa ${n.layer+1} · unidad ${n.index} · tanh ${fmt(value,3)}`)+` · ${n.silenced?'silenciada permanentemente':Math.abs(value)<.1?'baja actividad en este paso':'no lesionada'}`;}
  else $('unitDetail').textContent='Tocá una unidad para ver su capa, activación y estado.';
}
function drawWorld(world){
  const {ctx,width,height}=canvasContext('world'),view=worldTransform(world,width,height),{scale,ox,oy,x,y}=view;
  ctx.fillStyle='#111f20';ctx.fillRect(ox,oy,world.width*scale,world.height*scale);ctx.strokeStyle='#3b5652';ctx.strokeRect(ox,oy,world.width*scale,world.height*scale);
  ctx.save();ctx.beginPath();ctx.rect(ox,oy,world.width*scale,world.height*scale);ctx.clip();
  ctx.strokeStyle='#233936';ctx.lineWidth=.5;
  for(let i=1;i<10;i++){ctx.beginPath();ctx.moveTo(x(i),oy);ctx.lineTo(x(i),y(0));ctx.moveTo(ox,y(i));ctx.lineTo(x(world.width),y(i));ctx.stroke();}
  world.obstacles.forEach(o=>{ctx.fillStyle='#4a6066';ctx.fillRect(x(o.x),y(o.y+o.height),o.width*scale,o.height*scale);ctx.strokeStyle='#71838a';ctx.strokeRect(x(o.x),y(o.y+o.height),o.width*scale,o.height*scale);});
  world.foods.forEach(food=>{if(food.collected){ctx.strokeStyle='#516b44';ctx.beginPath();ctx.arc(x(food.x),y(food.y),food.radius*scale,0,Math.PI*2);ctx.stroke();return;}ctx.shadowColor='#b4e584';ctx.shadowBlur=12;ctx.fillStyle='#b4e584';ctx.beginPath();ctx.arc(x(food.x),y(food.y),food.radius*scale,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;});
  const f=world.fly;ctx.save();ctx.translate(x(f.x),y(f.y));ctx.rotate(view.rotation(f.heading));
  ctx.fillStyle='rgba(111,198,221,.07)';ctx.strokeStyle='rgba(111,198,221,.3)';ctx.beginPath();ctx.moveTo(0,0);ctx.arc(0,0,scale*4,-.7,.7);ctx.closePath();ctx.fill();ctx.stroke();
  ctx.fillStyle='#d9e3c7';ctx.beginPath();ctx.ellipse(-1,-3,7,3,-.45,0,Math.PI*2);ctx.ellipse(-1,3,7,3,.45,0,Math.PI*2);ctx.fill();ctx.fillStyle='#8fc9be';ctx.beginPath();ctx.ellipse(0,0,6,2.5,0,0,Math.PI*2);ctx.fill();ctx.fillStyle='#eef9d8';ctx.beginPath();ctx.arc(5,0,2.4,0,Math.PI*2);ctx.fill();ctx.restore();
  ctx.restore();ctx.fillStyle='#91aba4';ctx.font='10px monospace';ctx.fillText(`Alimento ${world.foods_collected}/3 · Choques ${world.collisions}${world.success?' · OBJETIVO CUMPLIDO':''}`,12,height-3);
}
function drawDecision(frame){
  const reader=ui.replay.decision;if(!reader)return;
  const decoded=decodeDecision(frame.h2,reader.weights,reader.bias),winner=frame.decision.logits.indexOf(Math.max(...frame.decision.logits)),valid=frame.action!==undefined;
  const chosen=valid?reader.actions[winner]:'Episodio terminado';
  $('decisionWinner').textContent=chosen;$('decisionTopology').textContent=`${frame.h2.length} → ${reader.actions.length} · lector PPO lineal`;
  $('actionScores').replaceChildren(...reader.actions.map((name,a)=>{const card=document.createElement('div');card.className='action-score'+(a===winner&&valid?' selected':'');const label=document.createElement('strong');label.textContent=name;const score=document.createElement('span');score.textContent=`Logit ${fmt(frame.decision.logits[a],3)} · ${fmt(frame.decision.probabilities[a]*100,1)}%`;const note=document.createElement('small');note.textContent=`Sesgo ${fmt(reader.bias[a],3)}${a===winner&&valid?' · ELEGIDA':''}`;card.append(label,score,note);return card;}));
  const ranked=topContributions(decoded[winner].contributions,6);
  $('decisionContributors').replaceChildren(...ranked.map(({index,value})=>{const li=document.createElement('li');const cell=ui.replay.cells?.[index];li.textContent=`${cell?cell.type+' · ID '+cell.id:'Unidad '+index}: ${value>=0?'+':''}${fmt(value,4)}`;li.style.color=value>=0?'#86dda7':'#eca35b';return li;}));
  const {ctx,width,height}=canvasContext('decisionNetwork'),n=frame.h2.length;
  const columns=4,rows=Math.ceil(n/columns),left=30,right=width-120,top=25,gap=(Math.min(width*.5,350)-left)/columns;
  const points=frame.h2.map((activity,i)=>({x:left+(i%columns)*gap,y:top+Math.floor(i/columns)*(height-50)/rows,activity}));
  const output=reader.actions.map((name,a)=>({x:right,y:height*(a+1)/(reader.actions.length+1),name}));
  output.forEach((o,a)=>{const chosen=a===winner&&valid;topContributions(decoded[a].contributions,12).forEach(({index,value})=>{const p=points[index];ctx.strokeStyle=value>=0?'rgba(134,221,167,.35)':'rgba(236,163,91,.35)';ctx.lineWidth=.5+Math.min(2,Math.abs(value)*3);ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.bezierCurveTo(width*.48,p.y,width*.62,o.y,o.x,o.y);ctx.stroke();});ctx.fillStyle=chosen?'#b4e584':'#7c969d';ctx.beginPath();ctx.arc(o.x,o.y,chosen?8:5,0,Math.PI*2);ctx.fill();ctx.font='11px monospace';ctx.textAlign='left';ctx.fillText(o.name,o.x+13,o.y-4);ctx.fillText(fmt(frame.decision.logits[a],3),o.x+13,o.y+12);});
  points.forEach((p,i)=>{const silent=!ui.replay.alive[i];ctx.fillStyle=silent?'#d38c7e':p.activity>=0?'#86dda7':'#eca35b';ctx.globalAlpha=silent?1:.2+.8*Math.abs(p.activity);ctx.beginPath();ctx.arc(p.x,p.y,2+Math.abs(p.activity)*2,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;if(i%12===0){ctx.fillStyle='#71878b';ctx.font='8px monospace';ctx.fillText(ui.replay.cells?.[i]?.type||String(i),p.x+7,p.y+3);}});
  ctx.fillStyle='#839195';ctx.font='9px monospace';ctx.fillText(`${n} actividades · ${valid?'decisión ejecutada':'estado final, sin ejecutar acción'}`,left,12);
}
function drawChart() {
  const {ctx,width,height}=canvasContext('chart');
  if(!ui.data)return;
  const current=selection(), left=40,right=15,top=25,bottom=32;
  const healthy=ui.data.series.filter(p=>p.architecture===current.architecture&&p.seed===current.seed&&p.condition==='healthy');
  const damaged=ui.data.series.filter(p=>p.architecture===current.architecture&&p.seed===current.seed&&p.condition===current.condition);
  const max=ui.data.config.recovery_steps;
  const values=[...healthy,...damaged].map(p=>p.reward);
  const foraging=ui.data.config.environment==='foraging',minY=foraging?Math.floor(Math.min(0,...values)/20)*20:0,maxY=foraging?Math.max(minY+20,Math.ceil(Math.max(1,...values)/20)*20):500;
  const x=v=>left+(width-left-right)*v/max, y=v=>height-bottom-(height-top-bottom)*(v-minY)/(maxY-minY);
  ctx.font='9px -apple-system, sans-serif';ctx.textAlign='right';ctx.textBaseline='middle';
  for(let tick=0;tick<=5;tick++){const value=minY+(maxY-minY)*tick/5;{ctx.strokeStyle='#29363e';ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(left,y(value));ctx.lineTo(width-right,y(value));ctx.stroke();ctx.fillStyle='#789097';ctx.fillText(fmt(value,0),left-10,y(value));}}
  ctx.textAlign='center';
  healthy.forEach(p=>{ctx.fillStyle='#789097';ctx.fillText(integer(p.steps),x(p.steps),height-13);});
  const path=(points,color,dashed)=>{
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dashed?[5,5]:[]);ctx.beginPath();
    points.forEach((p,i)=>i?ctx.lineTo(x(p.steps),y(p.reward)):ctx.moveTo(x(p.steps),y(p.reward)));ctx.stroke();ctx.setLineDash([]);
    if(!dashed){points.forEach(p=>{ctx.fillStyle=color;ctx.beginPath();ctx.arc(x(p.steps),y(p.reward),3.1,0,7);ctx.fill();});}
  };
  path(healthy,'#789dad',true);path(damaged,'#b4e584',false);
}
function render() {
  if(!ui.replay)return;
  const index=frameIndex(ui.frame, ui.replay.frames.length);
  const frame=ui.replay.frames[index], end=index>=ui.replay.frames.length-1;
  $('timeline').value=index;$('stepValue').textContent=frame.step;
  $('episodeScore').textContent=`${frame.step} / ${ui.replay.frames.length-1}`;
  $('replayStatus').textContent=end?(ui.replay.truncated?'Límite del episodio alcanzado':'Episodio finalizado'):ui.playing?'Reproduciendo episodio real':'Episodio real · en pausa';
  $('positionValue').textContent=frame.world?`${fmt(frame.world.fly.x,1)}, ${fmt(frame.world.fly.y,1)}`:fmt(frame.state[0],2);$('angleValue').textContent=fmt((frame.world?frame.world.fly.heading:frame.state[2])*180/Math.PI,1);
  $('actionLabel').textContent=frame.action===undefined?'Acción: episodio terminado':`Acción: ${frame.world?['girar izquierda','avanzar','girar derecha'][frame.action]:frame.action?'derecha →':'← izquierda'}`;
  drawBrain(frame);drawWorld(frame.world);drawDecision(frame);
}
function togglePlay() {
  if(!ui.replay)return;
  if(ui.frame>=ui.replay.frames.length-1)ui.frame=0;
  ui.playing=!ui.playing;ui.last=0;
  $('play').textContent=ui.playing?'Ⅱ':'▶';$('play').setAttribute('aria-label',ui.playing?'Pausar episodio':'Reproducir episodio');render();
}
function animate(now) {
  if(brainView.orbit&&ui.replay&&now-orbitLast>100){brainView.yaw+=.012;orbitLast=now;brainView.draw();}
  if(ui.playing && ui.replay) {
    if(ui.last)ui.frame+=Math.min(now-ui.last,100)/20*Number($('speed').value);
    ui.last=now;
    if(ui.frame>=ui.replay.frames.length-1){ui.frame=ui.replay.frames.length-1;ui.playing=false;$('play').textContent='▶';$('play').setAttribute('aria-label','Reproducir episodio');}
    render();
  }
  requestAnimationFrame(animate);
}
$('unitSelect').addEventListener('change',()=>{if(!ui.replay)return;selectedUnit=$('unitSelect').value===''?null:brainView.graph.nodes[Number($('unitSelect').value)];if(selectedUnit)brainView.module=selectedUnit.module;render();});
$('focusBrain').addEventListener('click',()=>{brainView.focus=!brainView.focus;$('focusBrain').textContent=brainView.focus?'Ver atlas completo':'Enfocar circuito';$('focusBrain').setAttribute('aria-pressed',brainView.focus);render();});
$('orbitBrain').addEventListener('click',()=>{brainView.orbit=!brainView.orbit;$('orbitBrain').textContent=brainView.orbit?'Órbita encendida':'Órbita apagada';$('orbitBrain').setAttribute('aria-pressed',brainView.orbit);});
$('rotateBrain').addEventListener('click',()=>{brainView.yaw+=.4;render();});
$('resetBrain').addEventListener('click',()=>{brainView.yaw=.12;brainView.pitch=-.12;render();});
$('showDamage').addEventListener('click',()=>{brainView.damage=!brainView.damage;$('showDamage').setAttribute('aria-pressed',brainView.damage);render();});
$('allModules').addEventListener('click',()=>{brainView.module=null;selectedUnit=null;render();});
document.querySelectorAll('[data-module]').forEach(button=>button.addEventListener('click',()=>{brainView.module=Number(button.dataset.module);selectedUnit=null;render();}));
for(const id of controls)$(id).addEventListener('change',loadReplay);
document.querySelectorAll('[data-stage]').forEach(button=>button.addEventListener('click',()=>{ui.stage=button.dataset.stage;loadReplay();}));
$('play').addEventListener('click',togglePlay);
$('restart').addEventListener('click',()=>{ui.playing=false;ui.frame=0;$('play').textContent='▶';$('play').setAttribute('aria-label','Reproducir episodio');render();});
$('timeline').addEventListener('input',()=>{if(!ui.replay)return;ui.frame=frameIndex(Number($('timeline').value),ui.replay.frames.length);ui.playing=false;$('play').textContent='▶';$('play').setAttribute('aria-label','Reproducir episodio');render();});
$('toggleEdges').addEventListener('click',()=>{ui.edges=!ui.edges;$('toggleEdges').setAttribute('aria-pressed',ui.edges);render();});
$('presentation').addEventListener('click',()=>{const enabled=document.body.classList.toggle('presentation');$('presentation').textContent=enabled?'Salir de presentación ↙':'Presentar ↗';setTimeout(()=>{render();drawChart();},50);});
for(const id of ['about','help','helpMobile'])$(id).addEventListener('click',()=>$('info').showModal());
for(const id of ['closeInfo','understood'])$(id).addEventListener('click',()=>$('info').close());
$('info').addEventListener('click',event=>{if(event.target===$('info')){const box=$('info').getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)$('info').close();}});
$('download').addEventListener('click',()=>{if(!ui.data)return;const url=URL.createObjectURL(new Blob([JSON.stringify(ui.data,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`flyheal_${ui.data.run}_datos.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
window.addEventListener('resize',()=>{render();drawChart();});
window.addEventListener('keydown',event=>{if(event.code==='Space'&&!['INPUT','SELECT','BUTTON','TEXTAREA'].includes(document.activeElement.tagName)&&!$('info').open){event.preventDefault();togglePlay();}});
const commitRun = latestLoader(run=>json('/api/summary?'+new URLSearchParams({run})), async data=>{
    ui.data=data;
    const foraging=data.config.environment==='foraging',evaluations=data.config.eval_episodes;
    $('environmentPanel').classList.toggle('foraging',foraging);$('sensoryNote').hidden=!foraging;
    $('environmentName').textContent=foraging?'Mosca · mundo 2D':'Mosca · mundo 2D';
    $('environmentMeta').textContent=foraging?'FLYHEAL / MUNDO 2D':'FLYHEAL / MUNDO 2D';
    $('positionUnit').textContent=foraging?'u.':'m';$('positionCaption').textContent=foraging?'POSICIÓN X,Y':'POSICIÓN';$('angleCaption').textContent=foraging?'ORIENTACIÓN':'ÁNGULO';$('episodeUnit').textContent=foraging?'recompensa':'/500';
    document.querySelectorAll('.return-unit').forEach(el=>el.textContent=foraging?'recompensa':'/ 500');
    $('evaluationCount').textContent=`Evaluación reservada · ${evaluations} episodios por punto`;$('meanCount').textContent=`Retorno medio de ${evaluations} episodios`;
    $('chartFoot').textContent=`Media de ${evaluations} episodios por checkpoint · semilla seleccionada${foraging?' · recompensa con signo':' · máximo 500'}`;
    const names={structured:'Estructurada sintética',dense:'Densa',random:'Aleatoria',male_cns:'MaleCNS medido',rewired_male_cns:'MaleCNS recableado'};
    setOptions($('architecture'),ui.data.config.architectures.map(arch=>[arch,names[arch]]),ui.data.config.architectures.includes('male_cns')?'male_cns':'structured');
    setOptions($('seed'),ui.data.config.seeds.map(seed=>[seed,'Semilla '+seed]),0);
    setOptions($('condition'),ui.data.conditions.map(condition=>{const [mode,fraction]=condition.split('_');return [condition,`${fmt(Number(fraction)*100,0)}% de ${mode==='neurons'?'unidades':'conexiones'}`];}),'neurons_0.5');
    $('runName').textContent=ui.data.run.toUpperCase();$('replicateCount').textContent=ui.data.config.seeds.length;
    $('recoveryBudget').textContent=integer(ui.data.config.recovery_steps);
    $('runFooter').textContent='Corrida '+ui.data.run;
    for(const id of controls)$(id).disabled=false;
    await loadReplay();
});
async function loadRun(){
  ++ui.request; ui.playing=false; ui.replay=null; ui.data=null;
  $('play').disabled=true; $('timeline').disabled=true;
  for(const key of ['pre','post','recovered','healthy'])$(key+'Metric').textContent='—';
  for(const id of controls)$(id).disabled=true;
  if(!$('runSelect').value){$('replayStatus').textContent='Sin modelo entrenado · iniciá una nueva corrida';return;}
  $('replayStatus').textContent='Cargando corrida…';
  try{await commitRun($('runSelect').value);}catch(error){$('replayStatus').textContent=error.message;}
}
const architectureNames={structured:'Estructurada',random:'Aleatoria',dense:'Densa',male_cns:'MaleCNS medido',rewired_male_cns:'MaleCNS recableado'};
let trainingJob = null, pollTimer = null;
async function post(url, value={}) {
  const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
  const result=await response.json();
  if(!response.ok)throw new Error(result.error || 'No se pudo iniciar el entrenamiento');
  return result;
}
async function refreshRuns(selected) {
  const data=await json('/api/runs');
  const current=selected || $('runSelect').value || data.default;
  $('emptyState').hidden=data.runs.length>0;
  $('resultsView').hidden=data.runs.length===0;
  $('laboratory').hidden=data.runs.length===0;
  if(!data.runs.length){$('trainingPanel').open=true;$('replicateCount').textContent='0';}
  $('runSelect').disabled=data.runs.length===0;
  setOptions($('runSelect'),data.runs.map(run=>[run.id,`${run.environment==='foraging'?'Mundo 2D':'Mundo 2D'} · ${run.id} · ${run.width} unidades de ancho · ${run.seeds.length} semilla(s)`]),current);
}
function previewTraining() {
  const real=['male_cns','rewired_male_cns'].includes($('trainArchitecture').value),dense=$('trainArchitecture').value==='dense';
  if(real){$('trainWidth').value='80';$('trainEnvironment').value='foraging';$('trainDensity').value='0.2025';}else if($('trainDensity').value==='0.2025')$('trainDensity').value='0.3125';
  $('trainWidth').disabled=real;$('trainEnvironment').disabled=real;$('trainDensity').disabled=dense||real;
  const width=Number($('trainWidth').value),degree=dense?width:Math.max(1,Math.round(width*Number($('trainDensity').value)));
  $('trainingPreview').textContent=real?'80 neuronas del subgrafo validado · 1.296 conexiones anatómicas. Tamaño y topología fijos; se entrenan ganancias de conexiones supervivientes. El control recableado conserva grados y contactos salientes.':`${integer(width*2)} neuronas en dos capas · ${integer(width*degree)} conexiones en la segunda capa · ${degree} entradas por neurona. La primera capa conecta las ${$('trainEnvironment').value==='foraging'?8:4} observaciones a todas sus unidades.`;
}
function showJob(job) {
  trainingJob=job;
  const running=job.status==='running';
  $('startTraining').disabled=running;
  $('cancelTraining').hidden=!running;
  $('viewTraining').hidden=job.status!=='complete';
  $('trainingProgress').hidden=job.status==='idle';
  if(job.status==='idle')return;
  const progress=job.progress || {};
  const labels={running:'Entrenando en CPU',complete:'Entrenamiento terminado',cancelled:'Entrenamiento cancelado',failed:'El entrenamiento falló'};
  $('trainingState').textContent=labels[job.status] || job.status;
  const percent=progress.percent || 0;
  $('trainingPercent').textContent=fmt(percent,0)+'%';$('trainingBar').value=percent;
  const details=[`${integer(progress.trained_steps || 0)} / ${integer(progress.target_steps || 0)} pasos`, ({preparing:'Preparando',baseline:'Aprendizaje inicial',healthy:'Control sano','neurons_0.5':'Recuperación de unidades','edges_0.5':'Recuperación de conexiones',complete:'Gráficos e informe guardados'})[progress.condition] || progress.condition || 'Preparando'];
  if(progress.seed!==undefined)details.push('semilla '+progress.seed);
  if(progress.last_return!==null && progress.last_return!==undefined)details.push('última media: '+fmt(progress.last_return));
  if(job.error)details.push(job.error);
  $('trainingDetail').textContent=details.join(' · ');
}
async function pollTraining() {
  clearTimeout(pollTimer);
  try {
    const previous=trainingJob?.status;
    const job=await json('/api/progress');showJob(job);
    if(job.status==='complete' && previous!=='complete'){await refreshRuns(job.id);await loadRun();}
    if(job.status==='running')pollTimer=setTimeout(pollTraining,2000);
  }catch(error){$('trainingState').textContent=error.message;pollTimer=setTimeout(pollTraining,4000);}
}
for(const id of ['trainEnvironment','trainArchitecture','trainWidth','trainDensity'])$(id).addEventListener('change',previewTraining);
$('startTraining').addEventListener('click',async()=>{
  $('startTraining').disabled=true;
  try{
    const job=await post('/api/train',{environment:$('trainEnvironment').value,architecture:$('trainArchitecture').value,width:Number($('trainWidth').value),density:Number($('trainDensity').value),budget:$('trainBudget').value,seeds:Number($('trainSeeds').value)});
    showJob(job);pollTraining();
  }catch(error){$('startTraining').disabled=false;$('trainingProgress').hidden=false;$('trainingState').textContent=error.message;}
});
$('cancelTraining').addEventListener('click',async()=>{try{showJob(await post('/api/cancel'));clearTimeout(pollTimer);}catch(error){$('trainingState').textContent=error.message;}});
$('viewTraining').addEventListener('click',async()=>{await refreshRuns(trainingJob.id);await loadRun();$('laboratory').scrollIntoView({behavior:'smooth'});});
$('runSelect').addEventListener('change',loadRun);
async function init(){try{previewTraining();await refreshRuns();await loadRun();await pollTraining();}catch(error){$('replayStatus').textContent=error.message;}}
init();requestAnimationFrame(animate);
