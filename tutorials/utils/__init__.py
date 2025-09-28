"""Unified utilities for EV charging tutorials.

This package provides a stable, discoverable API for notebooks:
from tutorials.utils import evaluate_policy, wrap_policy, show, ...
"""

# Common utilities
from .common import (
    wrap_policy,
    check_env_interface,
    wrap_ma_policy,
    extract_from_obs,
    solve_cvx_with_fallback,
    create_ev_env,
    create_ma_ev_env,
    apply_wrappers,
    DemandChargeWrapper,
    PlanningEvalConfig,
    get_planning_eval_env_fn,
)

# External helpers
from .external import (
    sb3_make_vec_env,
    sb3_make_model,
    sb3_policy_fn,
    rllib_register_parallel_env,
    rllib_collect_one_episode,
    make_omnisafe_ready_env,
    safepo_import_single_agent,
    safepo_import_multi_agent,
    build_share_obs_dict,
)

# EV-specific utilities
from .ev import (
    evaluate_policy,
    evaluate_policy_batch,
    rollout_to_trajectory,
    collect_trajectory_direct,
    seek_to_active,
    traj_to_timeseries_df,
    standardize_trajectory_costs,
    sweep_noise,
    mpc_horizon_sweep,
    validate_mpc_config,
    save_evaluation_results,
    load_evaluation_results,
    cache_path,
    load_df_gz,
    save_df_gz,
    plot_robustness_heatmap,
    plot_action_distribution,
    plot_action_saturation,
    plot_actions_heatmap,
    plot_fairness,
    plot_agent_coordination_matrix,
    plot_critical_timeline,
    plot_algorithm_comparison_matrix,
    plot_training_curves,
    plot_dual_dynamics,
    plot_cost_return_pareto,
    plot_violation_heatmap,
    create_training_logger,
    get_cache_dir,
    ensure_dir,
    save_timeseries,
    load_timeseries,
    save_dual_log,
    load_dual_log,
    save_model_checkpoint,
    load_model_checkpoint,
    ExperimentTracker,
    run_scenario_sweep,
    run_multiagent_scenario_sweep,
    evaluate_multi_agent_policy,
    collect_ma_trajectory,
    save_marl_traj,
    plot_ma_training_comparison,
    plot_safety_metrics,
    SafeRLExperimentSuite,
    find_critical_windows_from_series,
)

# Baseline algorithms (adapted from SustainGym, using local envs)
from .baselines import (
    GreedyAlgorithm,
    RandomAlgorithm,
    MPC,
    OfflineOptimal,
)

# Notebook utilities
from .notebook import show, show_metrics, quick_plot, set_compact_notebook_style

# RL utilities
from .rl import (
    DictFlatteningWrapper,
    RunningMeanStd,
    layer_init,
    explained_variance,
    make_vec_envs,
    get_obs_shape,
    get_action_dim,
    set_random_seed,
    RolloutBuffer,
    LinearSchedule,
    # New RL helpers for action scaling and monitoring
    get_action_scale_and_bias,
    scale_action,
    unscale_action,
    compute_log_prob_with_squashing,
    TrainingMonitor,
    # Policy builders
    build_ppo_policy,
    build_sac_policy,
    build_lstm_policy,
    build_grpo_policy,
)

# MORL utilities
from .morl import (
    GradientConflictResolver,
    compute_surrogate_gradients,
    flatten_gradients,
    unflatten_gradients,
)

# MARL utilities
from .marl import (
    build_central_state,
    ValueNormalizer,
)

# Safe RL utilities
from .safe_rl import (
    BaseSafeRLTrainer,
    SafeRolloutBuffer,
    LagrangianMultiplier,
    create_safe_policy_wrapper,
    create_safe_ev_env,
    SafetyTrackingWrapper,
    pretrain_cost_value_function,
)

# Uncertainty wrappers
# Wrappers (now consolidated in common.py)
from .common import (
    DepartureUncertaintyWrapper,
    GrowingForecastErrorWrapper,
    ConstraintTighteningWrapper,
    CarbonEmphasisWrapper,
)

# Diagnostics
from .diagnostics import (
    pilot_signals_df,
    constraint_slack_t,
    constraint_slack_series,
    plot_slack_from_env,
    get_constraint_binding_stats,
    analyze_violation_patterns,
    test_safety_robustness,
    interactive_critical_timeline,
    policy_behavior_fingerprint,
    # Additional diagnostics re-exported for notebook convenience
    get_policy_diagnostics,
    compare_policies,
    algorithm_arena,
    analyze_dual_dynamics,
    explain_safety_decision,
    reconstruct_state,
)

# Shared models (for cross-notebook checkpoint loading)
from .rl import PPOActor
from .safe_rl import SafeActorCritic

__all__ = [
    # Common
    'wrap_policy','check_env_interface','wrap_ma_policy','extract_from_obs','solve_cvx_with_fallback','create_ev_env',
    'create_safe_ev_env','create_ma_ev_env','SafetyTrackingWrapper','apply_wrappers',
    'PlanningEvalConfig','get_planning_eval_env_fn',
    # EV core
    'evaluate_policy','evaluate_policy_batch','rollout_to_trajectory','sweep_noise','mpc_horizon_sweep','validate_mpc_config','get_cache_dir',
    'traj_to_timeseries_df','standardize_trajectory_costs','seek_to_active',
    'collect_trajectory_direct',
    'save_timeseries','load_timeseries','save_dual_log','load_dual_log','save_model_checkpoint','load_model_checkpoint',
    'ExperimentTracker','run_scenario_sweep','run_multiagent_scenario_sweep','evaluate_multi_agent_policy',
    # Baselines
    'GreedyAlgorithm','RandomAlgorithm','MPC','OfflineOptimal',
    # EV plotting
    'plot_robustness_heatmap','plot_action_distribution','plot_action_saturation','plot_actions_heatmap','plot_fairness',
    'plot_critical_timeline','find_critical_windows_from_series','plot_algorithm_comparison_matrix','plot_training_curves','plot_dual_dynamics','plot_cost_return_pareto','plot_violation_heatmap',
    'SafeRLExperimentSuite',
    # Notebook
    'show','show_metrics','quick_plot','set_compact_notebook_style',
    # RL
    'DictFlatteningWrapper','RunningMeanStd','layer_init','explained_variance','make_vec_envs','get_obs_shape','get_action_dim','set_random_seed','RolloutBuffer','LinearSchedule',
    'get_action_scale_and_bias','scale_action','unscale_action','compute_log_prob_with_squashing','TrainingMonitor',
    'build_ppo_policy','build_sac_policy','build_lstm_policy','build_grpo_policy',
    # MORL
    'GradientConflictResolver','compute_surrogate_gradients','flatten_gradients','unflatten_gradients',
    # MARL
    'build_central_state','ValueNormalizer',
    # Safe RL
    'BaseSafeRLTrainer','SafeRolloutBuffer','LagrangianMultiplier','create_safe_policy_wrapper',
    'DepartureUncertaintyWrapper','GrowingForecastErrorWrapper',
    'ConstraintTighteningWrapper','CarbonEmphasisWrapper',
    'pretrain_cost_value_function',
    # Diagnostics
    'pilot_signals_df','constraint_slack_t','constraint_slack_series','plot_slack_from_env',
    'get_constraint_binding_stats','analyze_violation_patterns','test_safety_robustness','interactive_critical_timeline','policy_behavior_fingerprint','reconstruct_state',
    'get_policy_diagnostics','compare_policies','algorithm_arena','analyze_dual_dynamics','explain_safety_decision',
    # External
    'sb3_make_vec_env','sb3_make_model','sb3_policy_fn','rllib_register_parallel_env','rllib_collect_one_episode',
    'make_omnisafe_ready_env','safepo_import_single_agent','safepo_import_multi_agent','build_share_obs_dict',
    # Misc utils
    'cache_path','load_df_gz','save_df_gz','ensure_dir','create_training_logger',
    # Models
    'PPOActor','SafeActorCritic',
]
