# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Configuration schema for driver service supporting multiple model backends."""

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class TrajectorySelectionConfig:
    """Selection between sampled trajectory candidates (Alpamayo models only).

    Only takes effect with ``ModelConfig.num_trajectory_samples > 1``.
    """

    # ALWAYS_FIRST, CLOSEST_3D or CLOSEST_LATERAL; the latter two keep the
    # driven path consistent with the plan of the previous cycle.
    strategy: str = "ALWAYS_FIRST"
    # Waypoints entering the distance average, counted from the first one used.
    max_num_distance_points: int = 64
    # Leading waypoints excluded from the distance average.
    skip_first_n_distance_points: int = 0


@dataclass
class VisualCacheConfig:
    """Reuse of vision-encoder outputs across frames (Qwen3-VL Alpamayo models).

    See ``alpasim_driver.visual_cache`` for the strategies.
    """

    enabled: bool = False
    # Cached images; must cover cameras x context frames x interleaved rollouts.
    max_entries: int = 64
    strategy: str = "exact"  # exact | near_duplicate
    # near_duplicate only: largest mean absolute difference of normalized
    # pixels (Qwen3-VL normalizes to [-1, 1]) for which features are reused.
    near_duplicate_threshold: float = 0.02
    # near_duplicate only: other frames one encoded frame may serve before the
    # encoder must run again.
    max_near_duplicate_uses: int = 2


@dataclass
class ModelConfig:
    """Unified model configuration for all model types.

    ``model_type`` is the entry-point name registered under the
    ``alpasim.models`` group (e.g. ``"alpamayo1"``, ``"alpamayo1_5"``, ``"vam"``,
    ``"transfuser"``, ``"manual"``).  The driver resolves it at runtime via the plugin
    registry, so new models can be added without changing driver code.

    Note: dtype is not exposed in config - each model hardcodes its expected dtype.
    VAM uses float16, Transfuser uses its TrainingConfig's torch_float_type.
    """

    model_type: str = MISSING  # Entry-point name in the alpasim.models registry
    checkpoint_path: str = MISSING  # Path to model checkpoint (.pt/.pth)
    device: str = MISSING  # Device to run inference on (cuda/cpu)
    tokenizer_path: str | None = None  # Only required for VAM
    # Weight for classifier-free guidance on the navigation instruction (A1.5
    # only).  Unset leaves guidance to the checkpoint's diffusion config; a
    # weight forces it on and blends the conditioned and unconditioned velocity
    # fields, which costs a second forward pass (~60 GB VRAM) and needs an
    # instruction to blend towards.
    cfg_guidance_weight: float | None = None
    force_determinism: bool = False  # Alpamayo models only
    num_trajectory_samples: int = 1  # Alpamayo models only
    # Where camera JPEGs are decoded.  ``cuda`` leaves the frames on the
    # device, so the model's preprocessing resizes there too (Alpamayo models
    # only; the others reach for numpy and will refuse a device frame).
    image_decode_device: str = "cpu"  # cpu | cuda
    # Alpamayo 2 only. Splits candidate sampling into sequential calls to cap
    # peak GPU memory; unset samples all candidates in one call.
    trajectory_candidate_microbatch_size: int | None = None
    trajectory_selection: TrajectorySelectionConfig = field(
        default_factory=TrajectorySelectionConfig
    )
    visual_cache: VisualCacheConfig = field(default_factory=VisualCacheConfig)


@dataclass
class InferenceConfig:
    """Inference configuration."""

    use_cameras: list[str] = MISSING
    max_batch_size: int = MISSING  # Maximum batch size for inference
    subsample_factor: int = 1
    context_length: int | None = None  # Override model's default context length
    output_frequency_hz: int = 10  # Frequency of trajectory decisions (Hz)


@dataclass
class RouteConfig:
    """Route and command configuration."""

    default_command: int = 2  # Default command: 0=right, 1=left, 2=straight
    use_waypoint_commands: bool = True  # Whether to interpret waypoints as commands
    command_distance_threshold: float = (
        2.0  # Lateral displacement threshold for command determination (meters)
    )
    min_lookahead_distance: float = (
        5.0  # Minimum distance to look ahead for waypoints (meters)
    )


@dataclass
class TrajectoryOptimizerConfig:
    """Trajectory optimization configuration. (VaVam-Eco)"""

    enabled: bool = False  # Whether to enable trajectory optimization

    # Optimization weights
    smoothness_weight: float = 1.0  # Weight for trajectory smoothness
    deviation_weight: float = 0.1  # Weight for deviation from original
    comfort_weight: float = 2.0  # Weight for comfort constraint penalty

    max_iterations: int = 100  # Maximum optimization iterations

    # Frenet retiming options
    retime_in_frenet: bool = True  # Whether to redistribute waypoints along path
    retime_alpha: float = 0.25  # Retiming strength [0,1]; higher = more front-loaded

    # Vehicle constraints
    max_deviation: float = 2.0  # Max deviation from original trajectory (meters)
    max_heading_change: float = 0.5236  # Max heading change (~30 degrees)
    max_speed: float = 15.0  # Maximum speed (m/s)
    max_accel: float = 5.0  # Maximum acceleration (m/s²)

    # Comfort limits (from PDMS spec)
    max_abs_yaw_rate: float = 0.95  # rad/s
    max_abs_yaw_acc: float = 1.93  # rad/s²
    max_lon_acc_pos: float = 4.89  # m/s²
    max_lon_acc_neg: float = -4.05  # m/s²
    max_abs_lon_jerk: float = 8.37  # m/s³


@dataclass
class SafetyValidatorConfig:
    """LiDAR/radar safety filter over the sampled trajectory candidates.

    Requires the runtime to forward LiDAR spins
    (``runtime.simulation_config.render_lidar_for_driver``), radar scans
    (``runtime.simulation_config.radars``), or both; see
    ``alpasim_driver.safety_filter``.  It can only switch between candidates
    when ``model.num_trajectory_samples > 1``; otherwise it only slows down.
    """

    enabled: bool = False
    lane_half_width: float = 2.5  # m, path corridor half-width
    prediction_horizon: float = 60.0  # m, max forward distance considered
    reaction_time: float = 0.5  # s, delay before braking begins
    max_decel: float = 6.0  # m/s^2
    min_ttc: float = 2.0  # s, LOW_TTC threshold
    emergency_ttc: float = 1.0  # s, emergency-brake threshold
    safety_buffer: float = 2.0  # m, margin beyond the stopping distance
    min_confidence: float = 0.1  # detections below this are ignored
    steering_path_gain: float = 12.0  # path curvature sensitivity
    confidence_weight_power: float = 2.0  # confidence -> risk weight exponent
    rear_min_ttc: float = 2.0  # s
    rear_emergency_ttc: float = 0.8  # s
    rear_prediction_horizon: float = 30.0  # m behind ego
    lidar_min_range_m: float = 2.0  # closer returns are ego self-hits
    lidar_max_range_m: float = 80.0
    # Radar fusion (``runtime.simulation_config.radars``); see radar_fusion.
    radar_association_gate_m: float = 2.0  # radar <-> LiDAR obstacle distance
    radar_only_confidence: float = 0.5  # obstacles no LiDAR detection explains
    # Safety sensors per weather label, used when ``weather.enabled``: one of
    # none (skip the check), lidar, radar or fused.  ``unknown`` applies until
    # the first weather query.  Without weather labeling every forwarded sensor
    # is used.
    weather_sensor_policy: dict[str, str] = field(
        default_factory=lambda: {
            "unknown": "fused",
            "clear": "none",
            "rain": "fused",
            "fog": "radar",
            "snow": "radar",
        }
    )


@dataclass
class SceneLabelConfig:
    """One scene label (weather or scenario) from the model's own VQA answers.

    The model scores each answer to the label's question about the latest
    frame of ``camera``; the session's label is the most frequent of the last
    ``history`` answers.  See ``alpasim_driver.scene_labels``.
    """

    enabled: bool = False
    camera: str = "camera_front_wide_120fov"
    # Simulation time between queries; each costs one short model call.
    query_interval_s: float = 2.0
    history: int = 3


@dataclass
class RectificationTargetConfig:
    """Target pinhole parameters for rectifying a rendered camera."""

    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    resolution_hw: tuple[int, int]
    radial: tuple[float, ...] = ()
    tangential: tuple[float, ...] = ()
    thin_prism: tuple[float, ...] = ()

    # Rectify a larger canvas before cropping so pinhole distortion does not
    # discard pixels needed at the final image boundary.
    max_overscan_scale: float = 2.0
    safety_margin_px: int = 10


@dataclass
class DriverConfig:
    """Main driver configuration supporting multiple model backends."""

    # Logging level (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"

    # Model configuration
    model: ModelConfig = MISSING

    # Server configuration
    host: str = MISSING
    port: int = MISSING

    # Inference configuration
    inference: InferenceConfig = MISSING

    route: RouteConfig = field(default_factory=RouteConfig)

    trajectory_optimizer: TrajectoryOptimizerConfig = field(
        default_factory=TrajectoryOptimizerConfig
    )

    safety_validator: SafetyValidatorConfig = field(
        default_factory=SafetyValidatorConfig
    )

    # Weather picks the safety sensors (safety_validator.weather_sensor_policy);
    # the scenario label is logged and shown in videos.
    weather: SceneLabelConfig = field(default_factory=SceneLabelConfig)
    scenario: SceneLabelConfig = field(default_factory=SceneLabelConfig)

    # Output configuration
    output_dir: str = MISSING

    # If true, generates debug images in `output_dir`
    plot_debug_images: bool = False

    # Optional f-theta-to-pinhole compatibility path, keyed by camera ID.
    # NuRec presets leave this unset because NuRec renders pinhole directly.
    rectification: dict[str, RectificationTargetConfig] | None = None
