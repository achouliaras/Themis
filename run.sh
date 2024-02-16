seed=$RANDOM
sample=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
#python themis_pretrain.py  domain=MiniGrid env=UnlockPickup-v0 render_mode=human seed=$seed num_unsup_steps=0 num_train_steps=50000 num_interact=100 max_feedback=1000 reward_batch=2 reward_update=100 feed_type=$sample human_teacher=False agent.batch_size=256
#python themis_train.py  domain=MiniGrid env=UnlockPickup-v0 render_mode=human seed=$seed num_unsup_steps=0 num_train_steps=50000 num_interact=100 max_feedback=10000 reward_batch=2 reward_update=100 feed_type=$sample human_teacher=False agent.batch_size=256 debug=True learn_reward=True

#python themis_pretrain.py  domain=ALE env=MsPacman-v5 render_mode=rgb_array seed=$seed num_seed_steps=5000 num_unsup_steps=0 num_train_steps=50000 num_interact=3000 max_feedback=1000 reward_batch=2 reward_update=100 feed_type=$sample human_teacher=False agent.batch_size=256
python themis_train.py  domain=ALE env=MsPacman-v5 render_mode=human seed=$seed num_seed_steps=5000 num_unsup_steps=0 num_train_steps=50000 num_interact=4000 max_feedback=1000 reward_batch=10 reward_update=200 feed_type=$sample human_teacher=False agent.batch_size=256 debug=True learn_reward=True

#python themis_pretrain.py  domain=Control env=Humanoid-v4 render_mode=human seed=$seed num_unsup_steps=1000 num_train_steps=500000 num_interact=3000 max_feedback=1000 reward_batch=8 reward_update=100 feed_type=$sample human_teacher=False agent.batch_size=256
#python themis_train.py  domain=Control env=Humanoid-v4 render_mode=human seed=$seed num_unsup_steps=9000 num_train_steps=500000 num_interact=3000 max_feedback=1000 reward_batch=8 reward_update=100 feed_type=$sample human_teacher=True