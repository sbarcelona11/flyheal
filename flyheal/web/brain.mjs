/** Synthetic positions are illustrative; MaleCNS soma coordinates are measured. Activity is simulated. */
export const MODULE_COLORS=['134,221,167','111,198,221','180,158,238','235,183,123'];
export const MODULE_CENTERS=[[-.98,.02,0],[-.34,-.05,0],[.34,-.05,0],[.98,.02,0]];
export const MODULE_RADII=[[.27,.43,.25],[.42,.62,.35],[.42,.62,.35],[.27,.43,.25]];

export function moduleStats(frame,alive,cells=null){
  if(cells){return ['input','interneuron','output'].map((role,module)=>{
    const indices=cells.flatMap((cell,i)=>cell.role===role?[i]:[]),total=indices.length;
    return {module,total,silenced:indices.filter(i=>!alive[i]).length,low:indices.filter(i=>alive[i]&&Math.abs(frame.h2[i])<.1).length,meanAbs:total?indices.reduce((sum,i)=>sum+Math.abs(frame.h2[i]),0)/total:0};
  });}
  const width=alive.length, block=width/4;
  return Array.from({length:4},(_,module)=>{
    let sum=0,silenced=0,low=0;
    for(let i=module*block;i<(module+1)*block;i++){
      sum+=Math.abs(frame.h1[i])+Math.abs(frame.h2[i]);
      if(!alive[i])silenced++;
      if(Math.abs(frame.h1[i])<.1)low++;
      if(alive[i]&&Math.abs(frame.h2[i])<.1)low++;
    }
    return {module,total:2*block,silenced,low,meanAbs:sum/(2*block)};
  });
}
export function brainGraph(replay){
  if(replay.population==='single'){
    const width=replay.alive.length,cells=replay.cells,roles=['input','interneuron','output'];
    const middle=[0,1,2].map(axis=>cells.reduce((sum,cell)=>sum+cell.position[axis],0)/width);
    const extent=Math.max(...cells.flatMap(cell=>cell.position.map((value,axis)=>Math.abs(value-middle[axis]))))||1;
    const nodes=cells.map((cell,index)=>({layer:1,index,module:roles.indexOf(cell.role),cell,silenced:!replay.alive[index],position:[(cell.position[0]-middle[0])/extent,-(cell.position[1]-middle[1])/extent,-(cell.position[2]-middle[2])/extent]}));
    const edges=[],original=replay.original_mask||replay.mask;
    original.forEach((row,target)=>row.forEach((value,source)=>{if(value)edges.push({source,target,removed:!replay.mask[target][source]||!replay.alive[target]||!replay.alive[source]});}));
    return {nodes,edges,width,realCoordinates:true};
  }
  const width=replay.alive.length, block=width/4, nodes=[];
  for(let layer=0;layer<2;layer++)for(let i=0;i<width;i++){
    const module=Math.floor(i/block),index=i%block+layer*block,total=2*block;
    const vertical=1-2*(index+.5)/total, angle=index*Math.PI*(3-Math.sqrt(5));
    const radius=.68+.27*((index*17%23)/23), ring=Math.sqrt(1-vertical*vertical)*radius;
    const center=MODULE_CENTERS[module],shape=MODULE_RADII[module];
    nodes.push({layer,index:i,module,silenced:layer===1&&!replay.alive[i],
      position:[center[0]+Math.cos(angle)*ring*shape[0],center[1]+vertical*shape[1]*radius,center[2]+Math.sin(angle)*ring*shape[2]]});
  }
  const edges=[], original=replay.original_mask || replay.mask;
  original.forEach((row,target)=>row.forEach((value,source)=>{
    if(value)edges.push({source,target:width+target,removed:!replay.mask[target][source]||!replay.alive[target]});
  }));
  return {nodes,edges,width};
}

export class BrainView{
  constructor(canvas,onSelect){
    this.canvas=canvas;this.onSelect=onSelect;this.atlas=null;this.focus=false;this.orbit=false;this.yaw=.12;this.pitch=-.12;this.zoom=1;this.module=null;this.damage=true;this.edges=true;this.graph=null;this.points=[];
    let pointer=null,moved=false;
    canvas.addEventListener('pointerdown',event=>{pointer=[event.clientX,event.clientY];moved=false;canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener('pointermove',event=>{
      if(pointer){const dx=event.clientX-pointer[0],dy=event.clientY-pointer[1];if(Math.abs(dx)+Math.abs(dy)>1)moved=true;this.yaw+=dx*.008;this.pitch=Math.max(-.9,Math.min(.9,this.pitch+dy*.008));pointer=[event.clientX,event.clientY];this.draw();}
      else{const box=canvas.getBoundingClientRect(),hit=this.hit(event.clientX-box.left,event.clientY-box.top);canvas.title=hit?`${hit.cell?hit.cell.type+' · ID '+hit.cell.id:'Módulo '+ 'ABCD'[hit.module]+' · capa '+(hit.layer+1)+' · unidad '+hit.index} · tanh ${this.value(hit).toFixed(3)}${hit.silenced?' · silenciada permanentemente':''}`:'Arrastrá para girar · tocá una unidad para inspeccionarla';}
    });
    canvas.addEventListener('pointerup',event=>{if(!moved){const box=canvas.getBoundingClientRect(),hit=this.hit(event.clientX-box.left,event.clientY-box.top);if(hit)this.onSelect(hit);}pointer=null;});
    canvas.addEventListener('pointercancel',()=>{pointer=null;});
  }
  hit(x,y){let result=null,distance=12;for(const p of this.points){const d=Math.hypot(x-p.x,y-p.y);if(d<distance){distance=d;result=p.node;}}return result;}
  setReplay(replay){this.graph=brainGraph(replay);this.frame=null;this.points=[];}
  setAtlas(atlas){this.atlas=atlas;}
  value(node){return this.frame?(node.layer?this.frame.h2:this.frame.h1)[node.index]:0;}
  draw(frame){
    if(frame)this.frame=frame;
    const box=this.canvas.getBoundingClientRect(),width=box.width,height=box.height,dpr=window.devicePixelRatio||1;
    this.canvas.width=Math.round(width*dpr);this.canvas.height=Math.round(height*dpr);
    const ctx=this.canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
    if(!this.graph||!this.frame)return;
    const scale=Math.min(width*.34,height*.61)*this.zoom;
    const project=([x,y,z])=>{const rx=x*Math.cos(this.yaw)+z*Math.sin(this.yaw),rz=z*Math.cos(this.yaw)-x*Math.sin(this.yaw),ry=y*Math.cos(this.pitch)-rz*Math.sin(this.pitch),depth=y*Math.sin(this.pitch)+rz*Math.cos(this.pitch),perspective=3.8/(3.8-depth);return {x:width/2+rx*scale*perspective,y:height*.48+ry*scale*perspective,z:depth};};
    ctx.lineWidth=.6;
    for(let module=0;module<(this.graph.realCoordinates?0:4);module++){
      const center=MODULE_CENTERS[module],radii=MODULE_RADII[module],focused=this.module===null||this.module===module;
      ctx.strokeStyle=`rgba(${MODULE_COLORS[module]},${focused?.18:.04})`;
      for(let axis=0;axis<3;axis++)for(let latitude=-1;latitude<=1;latitude++){
        ctx.beginPath();
        for(let i=0;i<=70;i++){const t=i/70*Math.PI*2,p=center.slice(),axes=[0,1,2].filter(a=>a!==axis),r=Math.sqrt(1-(latitude*.42)**2);p[axis]+=latitude*.42*radii[axis];p[axes[0]]+=Math.cos(t)*radii[axes[0]]*r;p[axes[1]]+=Math.sin(t)*radii[axes[1]]*r;const q=project(p);i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y);}ctx.stroke();
      }
    }
    let coordinates=this.graph.nodes.map(node=>node.position);
    if(this.graph.realCoordinates&&this.atlas&&!this.focus){
      const pts=this.atlas.points,min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
      for(const p of pts)for(let a=0;a<3;a++){min[a]=Math.min(min[a],p[a+1]);max[a]=Math.max(max[a],p[a+1]);}
      const center=min.map((v,a)=>(v+max[a])/2),extent=Math.max(...min.map((v,a)=>(max[a]-v)/2));
      const normalize=p=>[(p[0]-center[0])/extent,-(p[1]-center[1])/extent,-(p[2]-center[2])/extent];
      ctx.fillStyle='rgba(160,179,185,.13)';
      for(const p of pts){const q=project(normalize(p.slice(1)));ctx.fillRect(q.x,q.y,.8,.8);}
      coordinates=this.graph.nodes.map(node=>normalize(node.cell.position));
    }
    const points=this.graph.nodes.map((node,i)=>({...project(coordinates[i]),node}));this.points=points;
    if(this.edges){
      const visible=this.graph.edges.filter(edge=>(this.damage||!edge.removed)&&(this.module===null||this.graph.nodes[edge.source].module===this.module||this.graph.nodes[edge.target].module===this.module));
      const stride=Math.max(1,Math.ceil(visible.length/650));
      for(let i=0;i<visible.length;i+=stride){const edge=visible[i],a=points[edge.source],b=points[edge.target],strength=Math.abs(this.value(a.node)*this.value(b.node));ctx.strokeStyle=edge.removed?'rgba(222,125,119,.17)':`rgba(${MODULE_COLORS[b.node.module]},${.035+strength*.19})`;ctx.setLineDash(edge.removed?[2,5]:[]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo((a.x+b.x)/2,(a.y+b.y)/2-16,a.x===b.x?b.x+.1:b.x,b.y);ctx.stroke();}
      ctx.setLineDash([]);
    }
    for(const p of points.slice().sort((a,b)=>a.z-b.z)){
      const node=p.node,focused=this.module===null||node.module===this.module,intensity=Math.abs(this.value(node));
      if(node.silenced){if(!this.damage)continue;ctx.strokeStyle=focused?'#e39287':'#654642';ctx.lineWidth=1.3;ctx.beginPath();ctx.moveTo(p.x-3,p.y-3);ctx.lineTo(p.x+3,p.y+3);ctx.moveTo(p.x+3,p.y-3);ctx.lineTo(p.x-3,p.y+3);ctx.stroke();}
      else{const radius=2.2+intensity*2;const color=this.graph.realCoordinates?(this.value(node)>=0?'134,221,167':'236,163,91'):MODULE_COLORS[node.module];ctx.fillStyle=`rgba(${color},${focused?.18+intensity*.82:.07})`;ctx.shadowColor=`rgb(${color})`;ctx.shadowBlur=focused&&intensity>.65?9:0;ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;}
    }
    ctx.font='10px monospace';ctx.textAlign='center';
    if(!this.graph.realCoordinates)MODULE_CENTERS.forEach((center,module)=>{const p=project([center[0],MODULE_RADII[module][1]+.10,0]);ctx.fillStyle=`rgb(${MODULE_COLORS[module]})`;ctx.fillText(`MÓDULO ${'ABCD'[module]}`,p.x,p.y);});
  }
}

/** Keep the simulator's Cartesian convention: increasing Y and angle turn left. */
export function worldTransform(world,width,height){
  const scale=Math.min((width-24)/world.width,(height-24)/world.height),ox=(width-world.width*scale)/2,oy=(height-world.height*scale)/2;
  return {scale,ox,oy,x:value=>ox+value*scale,y:value=>oy+(world.height-value)*scale,rotation:angle=>-angle};
}
