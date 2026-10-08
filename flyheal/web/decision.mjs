/** Exact algebra of the learned linear PPO actor, not a causal neural attribution. */
export function decodeDecision(activity,weights,bias){
  return weights.map((row,a)=>{const contributions=row.map((w,i)=>w*activity[i]);return {contributions,logit:contributions.reduce((sum,v)=>sum+v,bias[a]),bias:bias[a]};});
}
export function topContributions(values,limit=12){return values.map((value,index)=>({value,index})).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)||a.index-b.index).slice(0,limit);}
