seed=$RANDOM
sample=1
#python PEBBLE_pretrain.py  domain=MiniGrid env=DistShift1-v0 render_mode=human seed=$seed num_unsup_steps=1000 num_train_steps=500000 num_interact=3000 max_feedback=1000 reward_batch=8 reward_update=100 feed_type=$sample human_teacher=False agent.batch_size=256
#python PEBBLE_train.py  domain=MiniGrid env=DistShift1-v0 render_mode=human seed=$seed num_unsup_steps=1000 num_train_steps=500000 num_interact=3000 max_feedback=1000 reward_batch=8 reward_update=100 feed_type=$sample human_teacher=False agent.batch_size=256

python PEBBLE_pretrain.py  domain=ALE env=Breakout-v5 render_mode=human seed=$seed num_unsup_steps=1000 num_train_steps=500000 num_interact=3000 max_feedback=1000 reward_batch=8 reward_update=100 feed_type=$sample human_teacher=False agent.batch_size=256
#python PEBBLE_train.py  domain=ALE env=Breakout-v5 render_mode=human seed=$seed num_unsup_steps=1000 num_train_steps=500000 num_interact=3000 max_feedback=1000 reward_batch=8 reward_update=100 feed_type=$sample human_teacher=False agent.batch_size=256

#python PEBBLE_pretrain.py  domain=Control env=Humanoid-v4 render_mode=human seed=$seed num_unsup_steps=1000 num_train_steps=500000 num_interact=3000 max_feedback=1000 reward_batch=8 reward_update=100 feed_type=$sample human_teacher=False agent.batch_size=256
#python PEBBLE_train.py  domain=Control env=Humanoid-v4 render_mode=human seed=$seed num_unsup_steps=9000 num_train_steps=500000 num_interact=3000 max_feedback=1000 reward_batch=8 reward_update=100 feed_type=$sample human_teacher=True