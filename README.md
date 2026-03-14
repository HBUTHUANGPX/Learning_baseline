python scripts/train_hydra.py model=fsq data.motion_group=xsens_bvh data.history_frames=2 data.future_frames=8 train.epochs=200

python scripts/eval_mujoco_motion.py --mjcf /home/hpx/HPX_LOCO_2/mimic_baseline/general_motion_tracker_whole_body_teleoperation/general_motion_tracker_whole_body_teleoperation/assets/Q1/mjcf/Q1_wo_hand.xml --ckpt log/20260309_114301/checkpoint/epoch_200.pt --motion-feature-keys joint_pos,continuous_trigonometric_encoding,delta_yaw,root_height,root_xy_pos --use-isaac-to-mujoco-map

