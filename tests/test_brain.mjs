import {test} from 'node:test';
import assert from 'node:assert/strict';

test('module activity distinguishes low activation from permanent silence', async()=>{
  const {moduleStats}=await import('../flyheal/web/brain.mjs');
  const frame={h1:Array(16).fill(.2),h2:Array(16).fill(.4)};
  frame.h2[0]=0;frame.h2[1]=.02;
  const alive=Array(16).fill(1);alive[0]=0;
  const a=moduleStats(frame,alive)[0];
  assert.equal(a.total,8);assert.equal(a.silenced,1);assert.equal(a.low,1);
  assert.ok(Math.abs(a.meanAbs-(4*.2+.02+2*.4)/8)<1e-10);
});

test('brain graph shows only real effective edges and identifies actual damage', async()=>{
  const {brainGraph}=await import('../flyheal/web/brain.mjs');
  const original=Array.from({length:16},()=>Array(16).fill(0));
  original[0][3]=1;original[1][4]=1;original[2][5]=1;
  const mask=original.map(row=>row.slice());mask[1][4]=0;
  const alive=Array(16).fill(1);alive[2]=0;
  const graph=brainGraph({mask,original_mask:original,alive});
  assert.equal(graph.nodes.length,32);
  assert.deepEqual(graph.edges.filter(e=>!e.removed).map(e=>[e.source,e.target]),[[3,16]]);
  assert.equal(graph.edges.filter(e=>e.removed).length,2);
  assert.equal(graph.nodes.filter(n=>n.silenced).length,1);
  assert.ok(graph.nodes.every(n=>n.module>=0&&n.module<4));
});

test('world display preserves a left turn in Cartesian coordinates',async()=>{
  const {worldTransform}=await import('../flyheal/web/brain.mjs');
  const view=worldTransform({width:10,height:10},300,300);
  assert.ok(view.y(6)<view.y(5));
  assert.equal(view.rotation(Math.PI/4),-Math.PI/4);
  const heading=.35;
  assert.ok(view.y(5+Math.sin(heading)*.18)<view.y(5));
});

test('measured circuit uses one population and damage removes outgoing and incoming edges',async()=>{
  const {brainGraph,moduleStats}=await import('../flyheal/web/brain.mjs');
  const cells=[{id:11,role:'input',type:'LC4',position:[0,0,0]},{id:22,role:'interneuron',type:'test',position:[1,2,3]},{id:33,role:'output',type:'DNp01',position:[2,3,4]}];
  const mask=[[0,0,1],[1,0,0],[0,1,0]],alive=[1,0,1];
  const graph=brainGraph({population:'single',cells,mask,original_mask:mask,alive});
  assert.equal(graph.nodes.length,3);assert.equal(graph.edges.filter(e=>e.removed).length,2);
  assert.equal(graph.nodes[1].cell.id,22);
  const stats=moduleStats({h1:[8,8,8],h2:[.2,0,.4]},alive,cells);
  assert.equal(stats[0].total,1);assert.equal(stats[0].meanAbs,.2);
  assert.equal(stats[1].silenced,1);assert.equal(stats[2].meanAbs,.4);
});
