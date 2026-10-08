import test from 'node:test';import assert from 'node:assert/strict';
import {decodeDecision,topContributions} from '../flyheal/web/decision.mjs';
test('linear decision decomposes all signed contributions and bias',()=>{const result=decodeDecision([.5,-.25,0],[[2,4,9],[-1,2,3]],[.3,-.2]);assert.ok(Math.abs(result[0].logit-.3)<1e-10);assert.deepEqual(result[0].contributions,[1,-1,0]);assert.ok(Math.abs(result[1].logit+1.2)<1e-10);assert.deepEqual(topContributions([.1,-2,1],2).map(v=>v.index),[1,2]);});
