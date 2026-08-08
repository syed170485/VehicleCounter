"""
Lightweight centroid-based multi-object tracker.

This is intentionally simple (no Kalman filter, no re-ID embeddings) so the
whole pipeline stays readable and dependency-light. It works well for
traffic camera footage where vehicles move smoothly and don't overlap much
frame-to-frame. For denser/occluded scenes, swap this out for ByteTrack or
DeepSORT (both drop in easily since Ultralytics ships tracking support).
"""

from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=75):
        """
        Args:
            max_disappeared: frames an object can go undetected before its
                track is dropped.
            max_distance: max pixel distance between an existing track's
                centroid and a new detection for them to be considered the
                same object.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()       # object_id -> centroid (x, y)
        self.bboxes = OrderedDict()        # object_id -> (x1, y1, x2, y2)
        self.disappeared = OrderedDict()   # object_id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Args:
            detections: list of (x1, y1, x2, y2) bounding boxes for the
                current frame.

        Returns:
            OrderedDict of object_id -> (centroid, bbox) for currently
            tracked objects.
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._as_result()

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detections):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], detections[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # distance matrix: existing tracks x new detections
            distances = np.linalg.norm(
                object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :],
                axis=2,
            )

            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distances[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = detections[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(distances.shape[0])) - used_rows
            unused_cols = set(range(distances.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], detections[col])

        return self._as_result()

    def _as_result(self):
        return OrderedDict(
            (oid, (self.objects[oid], self.bboxes[oid])) for oid in self.objects
        )
