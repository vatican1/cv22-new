#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters
)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # давайе сначала получим облако 3d точек
    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]]) # посмотрели на соответствия 2d точек
    points_3d = triangulate_correspondences(correspondences,
                                pose_to_view_mat3x4(known_view_1[1]),
                                pose_to_view_mat3x4(known_view_2[1]),
                                intrinsic_mat,
                                TriangulationParameters(1000, 0, 0)
                                ) # трианглулировали соответствия

    # теперь у нас есть набор 3d точек, дальше имея знания о них необходимо искать положения камеры
    # Давайте сначала попробуем найти положение камеры в каждом 20 - ом кадре, а потом имея эту информацию будем уточнять положения во всех промежуточных кадрах

    # для начала напишем код просто для 20-ого кадра
    cv2.solvePnPRansac()


    # TODO: implement
    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count # пока что типо камера стоит
    corners_0 = corner_storage[0]
    point_cloud_builder = PointCloudBuilder(corners_0.ids[:1],
                                            np.zeros((1, 3)))

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
