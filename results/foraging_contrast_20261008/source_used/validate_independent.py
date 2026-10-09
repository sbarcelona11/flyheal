import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from flyheal.environments import ForagingEnv

torch.set_num_threads(1)
comparisons=[]
for label,checkpoint,encoding,steps in [('exploration_only','/app/results/foraging_contrast_20261008/validation_controls/exploration_102400.zip','raw',102400),('contrast_equal_budget','/app/results/foraging_contrast_20261008/validation_controls/contrast_102400.zip','contrast_v1',102400),('contrast_standard','/app/results/foraging_contrast_20261008/checkpoints/male_cns_seed0/baseline.zip','contrast_v1',153600)]:
    model=PPO.load(checkpoint,device='cpu');env=ForagingEnv(sensor_encoding=encoding);rows=[]
    for seed in range(300000,300020):
        obs,_=env.reset(seed=seed);total=0;counts=[0,0,0]
        for _ in range(500):
            action,_=model.predict(obs,deterministic=True);action=int(action);counts[action]+=1
            obs,reward,done,truncated,info=env.step(action);total+=reward
            if done or truncated:break
        rows.append(dict(map_seed=seed,**info,return_value=total,action_counts=counts))
    record=dict(label=label,training_seed=0,architecture='male_cns',training_steps=steps,sensor_encoding=encoding,ent_coef=.01,mean_foods=float(np.mean([r['foods_collected'] for r in rows])),success_rate=float(np.mean([r['success'] for r in rows])),mean_collisions=float(np.mean([r['collisions'] for r in rows])),episodes=rows)
    comparisons.append(record)
    print(json.dumps({k:v for k,v in record.items() if k!='episodes'}),flush=True)
    env.close()
Path('/app/results/foraging_contrast_20261008/independent_validation.json').write_text(json.dumps(dict(note='Separate held-out map seeds 300000..300019; one training seed, no cross-architecture claim',comparisons=comparisons),indent=2)+'\n')
