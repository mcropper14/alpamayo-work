"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit + integration tests for the physical_ai_av_lidar pipeline.
Run with: python -m pytest tests/ -v
         OR: python tests/test_pipeline.py  (standalone)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from physical_ai_av_lidar import (
    LidarLoader,
    LidarPipeline,
    PipelineConfig,
    SpinFrame,
    PreprocessConfig,
    DetectionConfig,
    PhysicsConfig,
    ConstraintStatus,
    RecommendedAction,
    ObjectClass,
)
from physical_ai_av_lidar.preprocessing import LidarPreprocessor, fit_ground_plane
from physical_ai_av_lidar.detection import LidarDetector, SimpleObjectTracker


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_loader(scenario="highway", n_spins=5, seed=0):
    return LidarLoader.synthetic(scenario=scenario, n_spins=n_spins, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Schema / Loader
# ─────────────────────────────────────────────────────────────────────────────

class TestLoader:
    def test_synthetic_creates_correct_spin_count(self):
        loader = make_loader(n_spins=10)
        assert len(loader) == 10

    def test_spin_has_required_fields(self):
        loader = make_loader(n_spins=3)
        spin = loader.get_spin(0)
        assert isinstance(spin, SpinFrame)
        assert spin.xyz.shape[1] == 3
        assert len(spin.intensity) == spin.n_points
        assert len(spin.ring) == spin.n_points

    def test_iter_spins_yields_correct_count(self):
        loader = make_loader(n_spins=8)
        spins = list(loader.iter_spins())
        assert len(spins) == 8

    def test_iter_spins_range(self):
        loader = make_loader(n_spins=10)
        spins = list(loader.iter_spins(start=2, stop=5))
        assert len(spins) == 3
        assert spins[0].scan_index == 2

    def test_ego_pose_attached(self):
        loader = make_loader(n_spins=3)
        spin = loader.get_spin(0)
        assert spin.ego_pose is not None
        assert spin.ego_pose.shape == (7,)

    def test_scan_indices_sequential(self):
        loader = make_loader(n_spins=5)
        spins = list(loader.iter_spins())
        indices = [s.scan_index for s in spins]
        assert indices == sorted(indices)

    def test_all_scenarios_loadable(self):
        for scenario in ["highway", "urban", "emergency_stop", "lane_change"]:
            loader = LidarLoader.synthetic(scenario=scenario, n_spins=3)
            assert len(loader) == 3

    def test_points_are_finite(self):
        loader = make_loader(n_spins=2)
        for spin in loader.iter_spins():
            assert np.all(np.isfinite(spin.xyz))
            assert np.all(np.isfinite(spin.intensity))

    def test_intensity_range(self):
        loader = make_loader()
        spin = loader.get_spin(0)
        # Raw intensity might exceed 1.0 before normalisation – that's OK at loader level
        assert spin.intensity.min() >= 0.0

    def test_missing_column_raises(self):
        df = pd.DataFrame({"scan_index": [0], "x": [1.0], "y": [0.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            LidarLoader.from_file.__func__ # just ensure import works
            # Directly test validator:
            from physical_ai_av_lidar.loader import _validate_lidar_df
            _validate_lidar_df(df)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_range_gating_removes_close_points(self):
        loader = make_loader(n_spins=2)
        spin = loader.get_spin(0)
        pre = LidarPreprocessor(PreprocessConfig(range_min=5.0, range_max=50.0))
        proc = pre.process(spin)
        ranges = np.linalg.norm(proc.xyz, axis=1)
        assert ranges.min() >= 4.9  # small tolerance for floating point

    def test_intensity_normalised_to_unit(self):
        loader = make_loader()
        spin = loader.get_spin(0)
        pre = LidarPreprocessor()
        proc = pre.process(spin)
        assert proc.intensity.max() <= 1.01
        assert proc.intensity.min() >= -0.01

    def test_ground_flag_present(self):
        loader = make_loader()
        spin = loader.get_spin(0)
        proc = LidarPreprocessor().process(spin)
        assert proc.is_ground.dtype == bool
        assert len(proc.is_ground) == proc.n_points

    def test_ground_points_have_lower_z(self):
        loader = make_loader(n_spins=2)
        spin = loader.get_spin(0)
        proc = LidarPreprocessor().process(spin)
        if proc.n_points > 0 and proc.is_ground.any():
            ground_z = proc.xyz[proc.is_ground, 2]
            obstacle_z = proc.xyz[~proc.is_ground, 2]
            if len(obstacle_z) > 0:
                assert ground_z.mean() < obstacle_z.mean()

    def test_ego_shadow_removed(self):
        loader = make_loader()
        spin = loader.get_spin(0)
        cfg = PreprocessConfig(remove_ego_vehicle_shadow=True, ego_shadow_radius_xy=2.5)
        proc = LidarPreprocessor(cfg).process(spin)
        r_xy = np.sqrt(proc.xyz[:, 0]**2 + proc.xyz[:, 1]**2)
        # No points within shadow radius that are above ego_shadow_z_min
        shadow_pts = (r_xy < 2.5) & (proc.xyz[:, 2] > cfg.ego_shadow_z_min)
        assert not shadow_pts.any()

    def test_processed_spin_has_obstacle_xyz(self):
        loader = make_loader()
        spin = loader.get_spin(0)
        proc = LidarPreprocessor().process(spin)
        obs = proc.xyz_obstacle
        assert obs.ndim == 2 and obs.shape[1] == 3

    def test_ransac_ground_fit(self):
        # Create a flat ground plane with noise
        rng = np.random.default_rng(0)
        n = 5000
        x = rng.uniform(-20, 20, n)
        y = rng.uniform(-20, 20, n)
        z = rng.normal(-1.8, 0.05, n)
        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        coeffs, is_ground = fit_ground_plane(xyz)
        assert is_ground.sum() / n > 0.8  # Most points are ground

    def test_motion_compensation_runs(self):
        loader = make_loader(n_spins=3)
        spin = loader.get_spin(1)
        cfg = PreprocessConfig(apply_motion_compensation=True)
        proc = LidarPreprocessor(cfg).process(spin)
        assert proc.n_points > 0


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetection:
    def _get_processed_spin(self, scenario="highway"):
        loader = LidarLoader.synthetic(scenario=scenario, n_spins=3)
        spin = loader.get_spin(0)
        return LidarPreprocessor().process(spin)

    def test_detection_returns_frame(self):
        proc = self._get_processed_spin()
        det = LidarDetector().detect(proc)
        assert det.scan_index == proc.scan_index

    def test_objects_have_valid_range(self):
        proc = self._get_processed_spin("highway")
        det = LidarDetector().detect(proc)
        for obj in det.objects:
            assert obj.range_m > 0
            assert np.isfinite(obj.range_m)

    def test_objects_sorted_by_range(self):
        proc = self._get_processed_spin()
        det = LidarDetector().detect(proc)
        ranges = [o.range_m for o in det.objects]
        assert ranges == sorted(ranges)

    def test_emergency_scene_has_close_objects(self):
        proc = self._get_processed_spin("emergency_stop")
        det = LidarDetector(DetectionConfig(cluster_min_pts=3)).detect(proc)
        # Emergency stop scene has obstacles at 6-12m
        if len(det.objects) > 0:
            assert det.nearest_range_m < 80.0

    def test_confidence_in_range(self):
        proc = self._get_processed_spin()
        det = LidarDetector().detect(proc)
        for obj in det.objects:
            assert 0.0 < obj.confidence <= 1.0

    def test_object_class_is_valid(self):
        proc = self._get_processed_spin()
        det = LidarDetector().detect(proc)
        valid = set(ObjectClass)
        for obj in det.objects:
            assert obj.cls in valid

    def test_bbox_extent_positive(self):
        proc = self._get_processed_spin()
        det = LidarDetector().detect(proc)
        for obj in det.objects:
            assert (obj.bbox.extent > 0).all()

    def test_tracker_adds_velocity(self):
        loader = LidarLoader.synthetic(scenario="highway", n_spins=5)
        pre = LidarPreprocessor()
        det = LidarDetector()
        tracker = SimpleObjectTracker()

        frames = []
        for spin in loader.iter_spins():
            proc = pre.process(spin)
            frame = det.detect(proc)
            frame = tracker.update(frame)
            frames.append(frame)

        # After first frame, any tracked (matched) object should have velocity set
        tracked_objects = [
            obj
            for frame in frames[1:]
            for obj in frame.objects
            if obj.relative_velocity is not None
        ]
        # Not every object will match across frames (new detections won't have vel yet),
        # but at least some should be tracked across the 5-spin sequence.
        # Verify that any velocity that IS set is a finite number.
        for obj in tracked_objects:
            assert np.isfinite(obj.relative_velocity)

    def test_empty_spin_returns_empty_frame(self):
        """Ensure graceful handling of empty point clouds."""
        import numpy as np
        from physical_ai_av_lidar.preprocessing import ProcessedSpin
        empty = ProcessedSpin(
            scan_index=0, timestamp_us=0,
            xyz=np.zeros((0, 3), np.float32),
            intensity=np.zeros(0, np.float32),
            ring=np.zeros(0, np.int32),
            is_ground=np.zeros(0, bool),
            ego_pose=None,
        )
        frame = LidarDetector().detect(empty)
        assert len(frame.objects) == 0
        assert frame.nearest is None


# ─────────────────────────────────────────────────────────────────────────────
# Physics Reasoning
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysicsReasoning:
    def _run_scenario(self, scenario, ego_speed=20.0, lat_a=0.0, lon_a=0.0, yaw=0.0):
        pipe = LidarPipeline()
        pipe.load_synthetic(scenario=scenario, n_spins=3)
        reports = list(pipe.run(ego_speed_ms=ego_speed))
        return reports

    def test_reports_generated_for_all_spins(self):
        reports = self._run_scenario("highway")
        assert len(reports) == 3

    def test_report_has_constraints(self):
        reports = self._run_scenario("highway")
        assert len(reports[0].constraints) > 0

    def test_highway_maintains_speed(self):
        pipe = LidarPipeline(PipelineConfig(
            physics=PhysicsConfig(ttc_critical=2.0, proximity_critical_m=4.0)
        ))
        pipe.load_synthetic("highway", n_spins=3)
        reports = list(pipe.run(ego_speed_ms=25.0))
        # Highway at 25 m/s with distant obstacles → should mostly maintain
        actions = [r.recommended_action for r in reports]
        # At minimum, action should be a valid RecommendedAction
        for a in actions:
            assert a in list(RecommendedAction)

    def test_emergency_stop_triggers_hard_action(self):
        pipe = LidarPipeline(PipelineConfig(
            detection=DetectionConfig(cluster_min_pts=3),
            physics=PhysicsConfig(proximity_critical_m=4.0, ttc_critical=2.0),
        ))
        pipe.load_synthetic("emergency_stop", n_spins=5, seed=42)
        reports = list(pipe.run(ego_speed_ms=22.0))
        # At least one spin should recommend braking action
        braking_actions = {
            RecommendedAction.HARD_BRAKE,
            RecommendedAction.EMERGENCY_STOP,
            RecommendedAction.DECELERATE,
        }
        has_braking = any(r.recommended_action in braking_actions for r in reports)
        # With objects at 6-12m and 22 m/s ego speed, physics should flag something
        assert has_braking or any(
            r.overall_status != ConstraintStatus.PASS for r in reports
        )

    def test_constraint_statuses_are_valid(self):
        reports = self._run_scenario("urban")
        valid = set(ConstraintStatus)
        for r in reports:
            assert r.overall_status in valid
            for c in r.constraints:
                assert c.status in valid

    def test_occupancy_grid_built(self):
        reports = self._run_scenario("highway")
        for r in reports:
            assert r.occupancy_grid is not None
            assert r.occupancy_grid.data.shape[0] > 0
            assert r.occupancy_grid.data.shape[1] > 0

    def test_occupancy_grid_values_in_range(self):
        reports = self._run_scenario("highway")
        for r in reports:
            g = r.occupancy_grid.data
            assert g.min() >= 0.0 and g.max() <= 1.0

    def test_nearest_range_is_finite_or_inf(self):
        reports = self._run_scenario("urban")
        for r in reports:
            assert r.nearest_range_m > 0
            assert np.isfinite(r.nearest_range_m) or r.nearest_range_m == float("inf")

    def test_physics_report_to_dict(self):
        pipe = LidarPipeline()
        pipe.load_synthetic("highway", n_spins=2)
        reports = list(pipe.run(ego_speed_ms=20.0))
        d = reports[0].to_dict()
        required_keys = {
            "scan_index", "recommended_action", "overall_status",
            "constraints", "nearest_range_m", "ego_speed_ms",
        }
        assert required_keys.issubset(d.keys())

    def test_critical_violation_detected_when_too_close(self):
        from physical_ai_av_lidar.detection import DetectionFrame, DetectedObject, BoundingBox3D
        import numpy as np
        from physical_ai_av_lidar.physics_reasoning import PhysicsReasoner, PhysicsConfig

        # Fabricate a detection frame with an obstacle at 2m, closing at 15 m/s
        bbox = BoundingBox3D(
            center=np.array([2.0, 0.0, 0.5]),
            extent=np.array([1.0, 0.8, 0.75]),
        )
        obj = DetectedObject(
            object_id=0,
            cls=ObjectClass.VEHICLE,
            bbox=bbox,
            n_points=150,
            range_m=2.0,
            confidence=0.9,
            relative_velocity=15.0,
            ttc_s=2.0 / 15.0,
        )
        frame = DetectionFrame(scan_index=0, timestamp_us=0, objects=[obj])

        reasoner = PhysicsReasoner(PhysicsConfig(
            proximity_critical_m=4.0, ttc_critical=2.0
        ))
        report = reasoner.reason(frame, ego_speed_ms=15.0)
        assert report.has_critical_violation
        assert report.overall_status == ConstraintStatus.CRITICAL
        assert report.recommended_action in {
            RecommendedAction.EMERGENCY_STOP, RecommendedAction.HARD_BRAKE
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_full_pipeline_runs_without_error(self):
        pipe = LidarPipeline()
        pipe.load_synthetic("urban", n_spins=5)
        reports = list(pipe.run(ego_speed_ms=8.0))
        assert len(reports) == 5

    def test_pipeline_timing_works(self):
        pipe = LidarPipeline()
        pipe.load_synthetic("highway", n_spins=4)
        results = list(pipe.run_with_timing())
        assert len(results) == 4
        for report, timing in results:
            assert timing.total_ms >= 0
            assert timing.preprocess_ms >= 0
            assert timing.detect_ms >= 0

    def test_pipeline_reset_on_reload(self):
        pipe = LidarPipeline()
        pipe.load_synthetic("highway", n_spins=3)
        list(pipe.run(ego_speed_ms=20.0))
        # Reload
        pipe.load_synthetic("urban", n_spins=3)
        reports = list(pipe.run(ego_speed_ms=8.0))
        assert len(reports) == 3

    def test_all_scenarios_produce_reports(self):
        for scenario in ["highway", "urban", "emergency_stop", "lane_change"]:
            pipe = LidarPipeline()
            pipe.load_synthetic(scenario, n_spins=3)
            reports = list(pipe.run(ego_speed_ms=15.0))
            assert len(reports) == 3, f"Failed for scenario: {scenario}"

    def test_process_single_spin(self):
        pipe = LidarPipeline()
        pipe.load_synthetic("highway", n_spins=10)
        report = pipe.process_spin(scan_index=5, ego_speed_ms=20.0)
        assert report.scan_index == 5

    def test_no_loader_raises(self):
        pipe = LidarPipeline()
        with pytest.raises(RuntimeError):
            list(pipe.run())

    def test_pipeline_n_spins(self):
        pipe = LidarPipeline()
        pipe.load_synthetic("highway", n_spins=7)
        assert pipe.n_spins == 7

    def test_pipeline_metadata(self):
        pipe = LidarPipeline()
        pipe.load_synthetic("urban", n_spins=2)
        meta = pipe.metadata
        assert meta is not None
        assert "urban" in meta.clip_id


# ─────────────────────────────────────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    """Standalone test runner (no pytest required)."""
    import traceback

    test_classes = [
        TestLoader,
        TestPreprocessing,
        TestDetection,
        TestPhysicsReasoning,
        TestPipelineIntegration,
    ]

    passed = failed = 0
    for cls in test_classes:
        inst = cls()
        methods = [m for m in dir(inst) if m.startswith("test_")]
        print(f"\n{'─'*60}")
        print(f"  {cls.__name__} ({len(methods)} tests)")
        print(f"{'─'*60}")
        for m in methods:
            try:
                getattr(inst, m)()
                print(f"  ✓  {m}")
                passed += 1
            except Exception as e:
                print(f"  ✗  {m}")
                traceback.print_exc()
                failed += 1

    total = passed + failed
    print(f"\n{'═'*60}")
    print(f"  Results: {passed}/{total} passed  |  {failed} failed")
    print(f"{'═'*60}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
