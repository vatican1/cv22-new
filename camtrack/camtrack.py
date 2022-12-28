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
    rodrigues_and_translation_to_view_mat3x4,
    view_mat3x4_to_rodrigues_and_translation,
    eye3x4,
)
from _corners import (
    CornerStorage,
    FrameCorners
)


def triangulate_nviews(mats, points):
    n = len(mats)
    solve_mat = np.zeros([3 * n, 4 + n])
    for i, (x, p) in enumerate(zip(points, mats)):
        solve_mat[3 * i:3 * i + 3, :4] = p
        solve_mat[3 * i:3 * i + 3, 4 + i] = -x
    A = np.linalg.svd(solve_mat)[-1][-1, :4]
    return A / A[3]


def try_add_3d_points(median_angel_, left: FrameCorners, right: FrameCorners, left_view_mat: np.ndarray,
                      right_view_mat: np.ndarray,
                      intrinsic_mat: np.ndarray):
    correspondences = build_correspondences(left, right)
    # print("Пытаюсь добавить 3d точки на кадрах:", left, right)
    if not correspondences:
        # print("Между кадрами нет соответствий")
        return False, None, None
    new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                 left_view_mat,
                                                                 right_view_mat,
                                                                 intrinsic_mat,
                                                                 TriangulationParameters(2.5, 5, 1)
                                                                 # 'max_reprojection_error', 'min_triangulation_angle_deg', 'min_depth'
                                                                 )
    if len(ids) != 0 and np.arccos(median_cos) > median_angel_:
        return True, new_points_3d, ids
    else:
        # print("триангуляция не решается с такими параметрами")
        if len(ids) == 0:
            pass
            # print("так как новых точек просто нет")
        if np.arccos(median_cos) < median_angel_:
            pass
            # print("так как не соблюдено условие на медианный угол, необходимый - ", median_angel_ / np.pi * 180,
            #       "получившийся - ", np.arccos(median_cos) / np.pi * 180)
        return False, None, None


def find_3d_2d_masks(ids_3d, ids_2d):
    mask3d = np.array([False] * len(ids_3d))
    mask2d = np.array([False] * len(ids_2d))
    for i, id_i in enumerate(ids_3d):
        for j, id_j in enumerate(ids_2d):
            if id_i == id_j:
                mask3d[i] = True
                mask2d[j] = True
    return mask3d, mask2d


def frame_to_add(storage_points_3d, storage_points_2d: FrameCorners, rvec_prev, tvec_prev, intrinsic_mat: np.ndarray,
                 amount_correspondences: int, repr_err=2.0):
    ids_3d = storage_points_3d[1]
    ids_2d = storage_points_2d.ids.flatten()

    mask3d, mask2d = find_3d_2d_masks(ids_3d, ids_2d)
    if mask2d.sum() < amount_correspondences:
        return False, None, None, None, None
    ids_3d_here = ids_3d[mask3d]
    points_3d, points_2d = storage_points_3d[0][mask3d], storage_points_2d.points[mask2d]

    # if rvec_prev is None:
    #     retval, r_vec, t_vec, inliers = cv2.solvePnPRansac(points_3d, points_2d, intrinsic_mat, np.array([]),
    #                                                        reprojectionError=repr_err,
    #                                                        iterationsCount=1000, confidence=0.99999)
    # else:
    retval, r_vec, t_vec, inliers = cv2.solvePnPRansac(points_3d, points_2d, intrinsic_mat, np.array([]),
                                                       reprojectionError=repr_err,
                                                       useExtrinsicGuess=True,
                                                       rvec=rvec_prev.copy(), tvec=tvec_prev.copy(),
                                                       iterationsCount=1000, confidence=0.99999)

    if retval:
        ids_outliers = []
        for i in inliers:
            if i not in range(len(ids_3d_here)):
                ids_outliers.append(ids_3d_here[i])
        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        return True, view_mat3x4_to_pose(view_mat), ids_outliers, r_vec, t_vec
    return False, None, None, None, None


def add_to_strorage(storage_points_3d, new_points_3d, ids):
    if len(ids) != 0:
        new_3d_points_mask = np.array([True if i not in storage_points_3d[1] else False for i in ids])
        if new_points_3d[new_3d_points_mask].shape[0] != 0:
            storage_points_3d[0] = np.vstack((storage_points_3d[0], new_points_3d[new_3d_points_mask]))
            storage_points_3d[1] = np.hstack((storage_points_3d[1], ids[new_3d_points_mask]))
            print("Облако 3d точек увеличилось до -", len(storage_points_3d[0]))
    return storage_points_3d


def find_nerest_frame(i, known_views):
    ret_ = known_views[0]
    min_ = abs(ret_ - i)
    for i in known_views:
        if abs(i - ret_) < min_:
            ret_ = i
    return ret_


def calculate_for_2_frames(intrinsic_mat: np.ndarray,
                           corner_storage: CornerStorage,
                           first_frame: int,
                           second_frame: int,
                           bound: int):
    correspondences = build_correspondences(corner_storage[first_frame],
                                            corner_storage[second_frame])

    if len(correspondences.ids) < bound:
        return False, None, None, None

    H, mask_homography = cv2.findHomography(correspondences.points_1, correspondences.points_2, cv2.RANSAC)

    E, mask_essential = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2,
                                             intrinsic_mat, cv2.RANSAC, 0.999, 1.0)

    if mask_homography.mean() > mask_essential.mean():
        return False, None, None, None

    essential_inliers_idx = np.arange(len(mask_essential))[mask_essential.flatten().astype(dtype=bool)]
    if len(correspondences.points_1[essential_inliers_idx]) < bound:
        return False, None, None, None

    retval, R, t, mask = cv2.recoverPose(E, correspondences.points_1[essential_inliers_idx],
                                         correspondences.points_2[essential_inliers_idx], intrinsic_mat)

    _, ids, median_cos = triangulate_correspondences(correspondences, eye3x4(),
                                                     pose_to_view_mat3x4(Pose(R.T, R.T @ -t)),
                                                     intrinsic_mat, TriangulationParameters(1.5, 9, 2))

    if len(ids) < bound:
        return False, None, None, None

    deg_angle = np.arccos(median_cos) / np.pi * 180
    metric = deg_angle  # - (first_frame + second_frame) / len(corner_storage) * 5
    return True, metric, R, t


def find_initial_frames(corner_storage,
                        intrinsic_mat,
                        n1=-1,
                        n2=-2):
    max_metric = 0
    n1 = -1
    n2 = -1
    R_0, t_0 = None, None
    bound = 80
    while n1 == -1 and bound > 10:
        print("считаю с ограничением в ", bound, "соответствий между кадрами")
        for i in range(0, len(corner_storage) // 2, 3):
            for j in range(i + 3, len(corner_storage) // 2, 3):
                retval, metric, R, t = calculate_for_2_frames(intrinsic_mat, corner_storage, i, j, bound)
                if retval and metric > max_metric:
                    n1, n2 = i, j
                    max_metric = metric
                    R_0 = R
                    t_0 = t
        bound //= 2

    print("инициализация произошла на кадрах:", n1, n2, " метрика - ", max_metric)
    return (n1, view_mat3x4_to_pose(eye3x4())), (n2, Pose(R_0.T, R_0.T @ -t_0))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = find_initial_frames(corner_storage, intrinsic_mat)

    known_views_numbers = [known_view_1[0], known_view_2[0]]
    known_views = {known_view_1[0]: known_view_1[1], known_view_2[0]: known_view_2[1]}
    r_vec1, t_vec_1 = view_mat3x4_to_rodrigues_and_translation(pose_to_view_mat3x4(known_view_1[1]))
    r_vec2, t_vec_2 = view_mat3x4_to_rodrigues_and_translation(pose_to_view_mat3x4(known_view_2[1]))
    known_r_vec_t_vec = {known_view_1[0]: (r_vec1, t_vec_1), known_view_2[0]: (r_vec2, t_vec_2)}

    shift = 10
    add_frames = 0
    if known_view_1[0] % shift != 0:
        add_frames += 1
    if known_view_2[0] % shift != 0:
        add_frames += 1

    # давайте сначала получим облако 3d точек
    correspondences = build_correspondences(corner_storage[known_view_1[0]],
                                            corner_storage[known_view_2[0]])  # посмотрели на соответствия 2d точек
    new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                 pose_to_view_mat3x4(known_view_1[1]),
                                                                 pose_to_view_mat3x4(known_view_2[1]),
                                                                 intrinsic_mat,
                                                                 TriangulationParameters(1.5, 4, 2)
                                                                 )  # сделали триангуляцию соответствия

    storage_points_3d = [new_points_3d, ids]  # тут храним все полученные 3d точки
    print("Облако 3d точек на инициализирующих кадрах: ", known_view_1[0], known_view_2[0], " - ",
          len(storage_points_3d[0]))

    # Идём по каждым 15 кадрам и какие-то добавляем
    know_views_len_prev = len(known_views_numbers)
    storage_points_3d_len_prev = len(storage_points_3d[1])
    flag_change = True
    iter = 0

    def_median_cos = np.pi / 40
    def_amount_correspondenses = 200

    median_cos = def_median_cos
    amount_correspondenses = def_amount_correspondenses
    while len(known_views_numbers) < len(corner_storage) // shift + add_frames:
        left_boarder = min(known_views_numbers)
        right_boarder = max(known_views_numbers)
        if not flag_change:
            print("ничего не изменилось!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
            median_cos /= 1.3
            if amount_correspondenses > 25:
                amount_correspondenses -= 15
        else:
            median_cos = def_median_cos
            amount_correspondenses = def_amount_correspondenses

        flag_change = False
        if iter >= 1:
            print("всё слишком плохо!")
            break
        iter += 1
        # пробуем найти новые известные положения камер
        print("размер облака - ", len(storage_points_3d[0]))
        for frame_number in range(0, len(corner_storage), shift):
            frame_nearest_number = find_nerest_frame(frame_number, known_views_numbers)
            retval, pose, _, r_v, t_v = frame_to_add(storage_points_3d, corner_storage[frame_number],
                                                     known_r_vec_t_vec[frame_nearest_number][0].copy(),
                                                     known_r_vec_t_vec[frame_nearest_number][1].copy(),
                                                     intrinsic_mat,
                                                     amount_correspondenses, 2.5)
            if retval:
                print("удалось получить положение камеры в кадре:", frame_number)
                if frame_number not in known_views_numbers:
                    known_views_numbers.append(frame_number)
                    known_views[frame_number] = pose
                    known_r_vec_t_vec[frame_number] = (r_v, t_v)
                else:
                    pass
                    # TODO взять среднее
                # добавились ли новые известные положения камер
                flag_change |= (know_views_len_prev != len(known_views_numbers))
                know_views_len_prev = len(known_views_numbers)
            else:
                print("при данном облаке точек не удалось получить положение камеры в кадре:", frame_number)
        # пробуем усилить наше облако 3d точек для всех пар
        known_views_numbers.sort()
        # print("пробуем увеличить облако 3d точек")
        for i1, left in enumerate(known_views_numbers):
            for j1, right in enumerate(known_views_numbers[::-1]):
                if len(known_views_numbers) - 1 - j1 <= i1:
                    break
                retval, new_points_3d, ids = try_add_3d_points(median_cos, corner_storage[left],
                                                               corner_storage[right],
                                                               pose_to_view_mat3x4(known_views[left]),
                                                               pose_to_view_mat3x4(known_views[right]),
                                                               intrinsic_mat)
                if retval:
                    storage_points_3d = add_to_strorage(storage_points_3d, new_points_3d, ids)
                    # добавились ли новые известные 3d точки
                    flag_change |= (storage_points_3d_len_prev != len(storage_points_3d[0]))
                    storage_points_3d_len_prev = len(storage_points_3d[0])
                    # print("облако 3d точек обновлено", len(storage_points_3d[0]), "кадры", left, right)
                else:
                    pass
                    # print("на кадрах:", left, right, "НЕ удалось увеличить облако 3d точек")

    inliers_storage_points_3d = np.array([True] * storage_points_3d[0].shape[0])
    view_mats = []
    for i in range(len(corner_storage)):

        storage_points_3d[0] = storage_points_3d[0][inliers_storage_points_3d]
        storage_points_3d[1] = storage_points_3d[1][inliers_storage_points_3d]

        frame_nearest_number = find_nerest_frame(i, known_views_numbers)
        retval, pose, ids_outliers, _, _ = frame_to_add(storage_points_3d, corner_storage[i],
                                                        known_r_vec_t_vec[frame_nearest_number][0],
                                                        known_r_vec_t_vec[frame_nearest_number][1],
                                                        intrinsic_mat, 4, 3)
        if not ids_outliers:
            pass
        else:
            for j in ids_outliers:
                inex_outlier = np.argwhere(storage_points_3d[1] == j)[0, 0]
                inliers_storage_points_3d[inex_outlier] = False
        if retval:
            print("для кадра", i, "УДАЛОСЬ решить PnP задачу")
            view_mats.append(pose_to_view_mat3x4(pose))
        else:
            retval, pose, ids_outliers, _, _ = frame_to_add(storage_points_3d, corner_storage[i],
                                                            known_r_vec_t_vec[frame_nearest_number][0],
                                                            known_r_vec_t_vec[frame_nearest_number][1],
                                                            intrinsic_mat,
                                                            4, 4)
            if retval:
                print("для кадра", i, "УДАЛОСЬ решить PnP задачу")
                view_mats.append(pose_to_view_mat3x4(pose))
            else:
                print("для кадра", i, "НЕ удалось решить PnP задачу")
                if len(view_mats) != 0:
                    view_mats.append(view_mats[-1])
                else:
                    frame_nearest_number = find_nerest_frame(i, known_views_numbers)
                    view_mats.append(pose_to_view_mat3x4(known_views[frame_nearest_number]))

    assert (len(corner_storage) == len(view_mats))
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
