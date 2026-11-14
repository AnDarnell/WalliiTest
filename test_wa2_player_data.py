import unittest

from wa2_player_data import _merge_cached_and_fetched_snapshots


class MergeCachedAndFetchedSnapshotsTests(unittest.TestCase):
    def test_preserves_cached_snapshots_and_prefers_fetched_overlaps(self):
        cached = [
            {"snapshot_time": "2026-08-06T10:00:00+00:00", "rating": 1000},
            {"snapshot_time": "2026-08-06T11:00:00+00:00", "rating": 1050},
        ]
        fetched = [
            {"snapshot_time": "2026-08-06T11:00:00+00:00", "rating": 1060},
            {"snapshot_time": "2026-08-06T12:00:00+00:00", "rating": 1100},
        ]

        merged = _merge_cached_and_fetched_snapshots(cached, fetched)

        self.assertEqual(
            merged,
            [
                {"snapshot_time": "2026-08-06T10:00:00+00:00", "rating": 1000},
                {"snapshot_time": "2026-08-06T11:00:00+00:00", "rating": 1060},
                {"snapshot_time": "2026-08-06T12:00:00+00:00", "rating": 1100},
            ],
        )


if __name__ == "__main__":
    unittest.main()
