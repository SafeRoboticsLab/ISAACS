from gym.envs.registration import register

register(
    id='spiritRL-v0',
    entry_point='spirit_rl.envs:SpiritRLEnv'
)

register(
    id='spiritRLDuo-v0',
    entry_point='spirit_rl.envs:SpiritRLEnvDuo'
)

register(
    id='spiritRLPerformance-v0',
    entry_point='spirit_rl.envs:SpiritRLEnvPerformance'
)

register(
    id='spiritRLPerformanceSAC-v0',
    entry_point='spirit_rl.envs:SpiritRLEnvPerformanceSAC'
)

register(
    id='spiritRLAdversarial-v0',
    entry_point='spirit_rl.envs:SpiritRLEnvAdversarial'
)