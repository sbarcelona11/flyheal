import {test} from 'node:test';
import assert from 'node:assert/strict';
import {frameIndex} from '../flyheal/web/playback.mjs';

test('a long previous replay cannot index past a shorter loaded episode', () => {
  assert.equal(frameIndex(214, 11), 10);
  assert.equal(frameIndex(4.8, 11), 4);
  assert.equal(frameIndex(-5, 11), 0);
  assert.equal(frameIndex(NaN, 11), 0);
});

test('older run response cannot replace the latest selected run', async () => {
  const {latestLoader}=await import('../flyheal/web/playback.mjs');
  let resolveA, resolveB, shown;
  const pending={A:new Promise(r=>resolveA=r),B:new Promise(r=>resolveB=r)};
  const load=latestLoader(name=>pending[name],result=>{shown=result;});
  const first=load('A'),second=load('B');
  resolveB('metrics B');await second;
  resolveA('metrics A');await first;
  assert.equal(shown,'metrics B');
});
