seed=$RANDOM
sample=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

python themis_pretrain.py device=mps \
       domain=ALE env=MsPacman-v5 render_mode=rgb_array seed=$seed \
       num_seed_steps=10000 num_unsup_steps=0 num_train_steps=1000000 \
       replay_buffer_capacity=1000000 reward_model_capacity=10000 \
       num_interact=4000 max_feedback=10000 reward_batch=50 reward_update=200 feed_type=$sample segment=120 \
       human_teacher=False debug=True learn_reward=True

python themis_train.py  device=mps \
       domain=ALE env=MsPacman-v5 render_mode=rgb_array seed=$seed \
       num_seed_steps=10000 num_unsup_steps=0 num_train_steps=1000000 \
       replay_buffer_capacity=1000000 reward_model_capacity=10000 \
       num_interact=4000 max_feedback=10000 reward_batch=50 reward_update=200 feed_type=$sample segment=120\
       human_teacher=False debug=True learn_reward=True \
       xplain_action=False xplain_state=False
