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

    def pnp_by_frame_number(points_3d, next_scene: int, info: bool):
        ids_3d = points_3d[1]  # id точек, для которых необходимо решать задачу pnp
        ids_2d = corner_storage[next_scene].ids
        intersect_ids = np.intersect1d(ids_3d, ids_2d)  # пересекли по id 3d и 2d точки
        mask_3d = np.in1d(ids_3d, intersect_ids)
        mask_2d = np.in1d(ids_2d, intersect_ids)
        if info:
            print("Кадр: ", next_scene, " количество соответствий: ", mask_2d[mask_2d].shape[0])
        return cv2.solvePnPRansac(points_3d[0][mask_3d],
                                  corner_storage[next_scene].points[mask_2d],
                                  intrinsic_mat,
                                  np.array([])
                                  )

    # давайте сначала получим облако 3d точек
    correspondences = build_correspondences(corner_storage[known_view_1[0]],
                                            corner_storage[known_view_2[0]])  # посмотрели на соответствия 2d точек
    new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                 pose_to_view_mat3x4(known_view_1[1]),
                                                                 pose_to_view_mat3x4(known_view_2[1]),
                                                                 intrinsic_mat,
                                                                 TriangulationParameters(1000, 0, 0)
                                                                 )  # сделали триангуляцию соответствия

    storage_points_3d = [new_points_3d, ids]  # тут храним все полученные 3d точки
    print("Облако 3d точек - ", len(storage_points_3d[0]))
    # будем идти по shift кадров и добавлять 3d точки
    # идём вправо
    shift = 15

    # if min(known_view_1[0], known_view_2[0]) - shift >= 0:
    #     left_frame_number = min(known_view_1[0], known_view_2[0])
    #     prev_view = known_view_1[1] if known_view_1[0] == left_frame_number else known_view_2[1]
    #     prev_view_mat = pose_to_view_mat3x4(prev_view)
    #     for i in range(left_frame_number - shift, -1, -shift):
    #         retval, r_vec, t_vec, inliers = pnp_by_frame_number(storage_points_3d, i, False)
    #         next_view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec,
    #                                                                  t_vec)
    #         correspondences = build_correspondences(corner_storage[left_frame_number], corner_storage[i])
    #         new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
    #                                                                      prev_view_mat,
    #                                                                      next_view_mat,
    #                                                                      intrinsic_mat,
    #                                                                      TriangulationParameters(1000, 0, 0)
    #                                                                      )
    #         new_3d_points_mask = np.array([True if i not in storage_points_3d[1] else False for i in ids])
    #         storage_points_3d[0] = np.vstack((new_points_3d[new_3d_points_mask], storage_points_3d[0]))
    #         storage_points_3d[1] = np.hstack((ids[new_3d_points_mask], storage_points_3d[1]))
    #         print("Облако 3d точек - ", len(storage_points_3d[0]))
    #         left_frame_number -= shift
    #         prev_view_mat = next_view_mat
    #
    #
    if max(known_view_1[0], known_view_2[0]) + shift < len(corner_storage):
        right_frame_number = max(known_view_1[0], known_view_2[0])
        prev_view = known_view_1[1] if known_view_1[0] == right_frame_number else known_view_2[1]
        prev_view_mat = pose_to_view_mat3x4(prev_view)
        for i in range(right_frame_number + shift, len(corner_storage), shift):
            # 1 - определяем позицию камеры на этом кадре
            # 2 - доставляем 3d точки, которые можем доставить
            retval, r_vec, t_vec, inliers = pnp_by_frame_number(storage_points_3d, i)
            next_view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec,
                                                                     t_vec)  # определили позицию камеры на данном кадре
            correspondences = build_correspondences(corner_storage[right_frame_number], corner_storage[i])
            new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                         prev_view_mat,
                                                                         next_view_mat,
                                                                         intrinsic_mat,
                                                                         TriangulationParameters(1000, 0, 0)
                                                                         )
            # добавим только новые 3d точки
            new_3d_points_mask = np.array([True if i not in storage_points_3d[1] else False for i in ids])
            storage_points_3d[0] = np.vstack((storage_points_3d[0], new_points_3d[new_3d_points_mask]))
            storage_points_3d[1] = np.hstack((storage_points_3d[1], ids[new_3d_points_mask]))
            right_frame_number += shift
            prev_view_mat = next_view_mat

    # сделать такой же проход в обратную сторону !!!!!

    # в итоге после этих операций у нас есть облако 3d точек, для которых можно решать pnp для каждого кадра

    view_mats = []
    for i in range(len(corner_storage)):
        retval, r_vec, t_vec, inliers = pnp_by_frame_number(storage_points_3d, i, True)

        view_mats.append(rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec))

    point_cloud_builder = PointCloudBuilder(storage_points_3d[1],  # id всех найденных 3d точек
                                            storage_points_3d[0])

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
