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
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4
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
    correspondences = build_correspondences(corner_storage[known_view_1[0]],
                                            corner_storage[known_view_2[0]])  # посмотрели на соответствия 2d точек
    points_3d = triangulate_correspondences(correspondences,
                                            pose_to_view_mat3x4(known_view_1[1]),
                                            pose_to_view_mat3x4(known_view_2[1]),
                                            intrinsic_mat,
                                            TriangulationParameters(1000, 0, 0)
                                            )  # трианглулировали соответствия

    # теперь у нас есть набор 3d точек, дальше имея знания о них необходимо искать положения камеры
    # Давайте сначала попробуем найти положение камеры в каждом 20 - ом кадре, а потом имея эту информацию будем уточнять положения во всех промежуточных кадрах

    # для начала напишем код просто для 20-ого кадра
    next_scene = 20  # тут надо написать код в духе не 20 кадр, а самый старший кадр из данных +20, если так нельзя то -20

    ids_3d = points_3d[1]  # id точек, для которых необходимо решать задачу pnp
    ids_2d = corner_storage[next_scene].ids
    intersect_ids = np.intersect1d(ids_3d, ids_2d)
    mask_3d = np.in1d(ids_3d, intersect_ids)
    mask_2d = np.in1d(ids_2d, intersect_ids)

    retval, r_vec, t_vec, inliers = cv2.solvePnPRansac(points_3d[0][mask_3d],
                                                     corner_storage[next_scene].points[mask_2d],
                                                     intrinsic_mat,
                                                     np.array([])
                                                     )
    new_view_mat_by_prev_position = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec) # получили view матрицу относительно предыдущей позиции камеры, теперь надо сделать её относительно начального положения
    new_view_mat = new_view_mat_by_prev_position # TODO написать что нужно чтобы получить view матрицу относительно начальног оположения камеры

    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count  # пока что типо камера стоит
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
