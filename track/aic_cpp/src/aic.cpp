#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cmath>

namespace py = pybind11;

py::array_t<double>
epipolar_3d_score_norm(const py::array_t<double> &pA,
                   const py::array_t<double> &rayA,
                   const py::array_t<double> &pB,
                   const py::array_t<double> &rayB,
                   double alpha_epi) {
    auto pA_buf = pA.unchecked<1>();  // Assuming 2D array
    auto rayA_buf = rayA.unchecked<2>(); // Assuming 2D array
    auto pB_buf = pB.unchecked<1>();  // Assuming 2D array
    auto rayB_buf = rayB.unchecked<2>(); // Assuming 2D array

    int num_points = rayA.shape(0);

    py::array_t<double> result({num_points});
    auto result_buf = result.mutable_unchecked<1>();

    for (int i = 0; i < num_points; ++i) {
        double cp[3];
        cp[0] = rayA_buf(i, 1) * rayB_buf(i, 2) - rayA_buf(i, 2) * rayB_buf(i, 1);
        cp[1] = rayA_buf(i, 2) * rayB_buf(i, 0) - rayA_buf(i, 0) * rayB_buf(i, 2);
        cp[2] = rayA_buf(i, 0) * rayB_buf(i, 1) - rayA_buf(i, 1) * rayB_buf(i, 0);
        double norm = sqrt(cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2]);
        double dot_product = (pA_buf(0) - pB_buf(0)) * cp[0] +
                             (pA_buf(1) - pB_buf(1)) * cp[1] +
                             (pA_buf(2) - pB_buf(2)) * cp[2];
        double dist = std::abs(dot_product) / (norm + 1e-6);
        result_buf(i) = 1 - dist / alpha_epi;
    }
    return result;
}

py::array_t<double> compute_joints_rays(const py::array_t<double>& keypoints_mv,
                                        const py::array_t<double>& cam_project_inv,
                                        const py::array_t<double>& cam_pos) {
    auto keypoints_mv_buf = keypoints_mv.unchecked<2>();  // Assuming 2D array
    auto cam_project_inv_buf = cam_project_inv.unchecked<2>();  // Assuming 2D array
    auto cam_pos_buf = cam_pos.unchecked<1>();  // Assuming 1D array

    int num_keypoints = keypoints_mv.shape(0);

    py::array_t<double> joints_rays_norm({num_keypoints, 3});

    auto joints_rays_norm_buf = joints_rays_norm.mutable_unchecked<2>();

    for (int i = 0; i < num_keypoints; i++) {
        // cam.project_inv[:, :2] @ self.keypoints_mv[v][:,:-1].T
        double joints_rays[4];
        for (int j = 0; j < 4; j++) {
            joints_rays[j] = cam_project_inv_buf(j, 0) * keypoints_mv_buf(i, 0)
                           + cam_project_inv_buf(j, 1) * keypoints_mv_buf(i, 1)
                           + cam_project_inv_buf(j, 2);
        }
        for (int j = 0; j < 3; j++) {
            joints_rays[j] = joints_rays[j] / joints_rays[3] - cam_pos_buf[j];
        }
        double norm = sqrt(joints_rays[0] * joints_rays[0] + joints_rays[1] * joints_rays[1] + joints_rays[2] * joints_rays[2]) + 1e-5;
        joints_rays_norm_buf(i, 0) = joints_rays[0] / norm;
        joints_rays_norm_buf(i, 1) = joints_rays[1] / norm;
        joints_rays_norm_buf(i, 2) = joints_rays[2] / norm;
    }

    return joints_rays_norm;
}

double aff_sum(const py::array_t<double>& aff_temp,
               const py::array_t<bool>& valid,
               const py::array_t<double>& age_2D_sv,
               double age_2D_thr) {

    auto aff_temp_buf = aff_temp.unchecked<1>();
    auto valid_buf = valid.unchecked<1>();
    auto age_2D_sv_buf = age_2D_sv.unchecked<1>();

    double aff = 0;
    double aff_norm = 1e-5;
    int num_keypoints = aff_temp.shape(0);

    for (int i=0; i<num_keypoints;i++){
        double w = valid_buf(i) * std::exp(-age_2D_sv_buf(i)) * (age_2D_sv_buf(i) <= age_2D_thr);
        aff += aff_temp_buf(i) * w;
        aff_norm += w;
    }
    return aff / aff_norm;
}

double compute_feet_distance(const py::array_t<double>& joints,
                             const std::vector<int> & feet_idxs,
                             const py::array_t<double>& homo_feet_inv,
                             const py::array_t<double>& feet_s,
                             double thred_homo) {
    auto joints_buf = joints.unchecked<2>();  // Assuming 2D array
    auto homo_feet_inv_buf = homo_feet_inv.unchecked<2>();  // Assuming 2D array
    auto feet_s_buf = feet_s.unchecked<1>();  // Assuming 1D array

    double feet_pos[2] = {0};
    for (auto &i: feet_idxs) {
        feet_pos[0] += joints_buf(i, 0);
        feet_pos[1] += joints_buf(i, 1);
    }
    // feet_idxs.size();

    double feet_t_[3];
    for (int j = 0; j < 3; j++) {
        feet_t_[j] = homo_feet_inv_buf(j, 0) * feet_pos[0] / feet_idxs.size()
                   + homo_feet_inv_buf(j, 1) * feet_pos[1] / feet_idxs.size()
                   + homo_feet_inv_buf(j, 2);
    }
    double f0 = feet_s_buf(0) - feet_t_[0] / feet_t_[2];
    double f1 = feet_s_buf(1) - feet_t_[1] / feet_t_[2];
    double norm = sqrt(f0 * f0 + f1 * f1);

    return 1 - norm / thred_homo;
}

py::array_t<double> compute_feet_s(const py::array_t<double>& joints,
                             const std::vector<int> & feet_idxs,
                             const py::array_t<double>& homo_feet_inv) {
    auto joints_buf = joints.unchecked<2>();  // Assuming 2D array
    auto homo_feet_inv_buf = homo_feet_inv.unchecked<2>();  // Assuming 2D array

    py::array_t<double> feet_s({3});
    auto feet_s_buf = feet_s.mutable_unchecked<1>();  // Assuming 1D array

    double feet_pos[2] = {0};
    for (auto &i: feet_idxs) {
        feet_pos[0] += joints_buf(i, 0);
        feet_pos[1] += joints_buf(i, 1);
    }
    // feet_idxs.size();

    double feet_t_[3];
    for (int j = 0; j < 3; j++) {
        feet_t_[j] = homo_feet_inv_buf(j, 0) * feet_pos[0] / feet_idxs.size()
                   + homo_feet_inv_buf(j, 1) * feet_pos[1] / feet_idxs.size()
                   + homo_feet_inv_buf(j, 2);
    }
    feet_s_buf(0) = feet_t_[0] / feet_t_[2];
    feet_s_buf(1) = feet_t_[1] / feet_t_[2];
    return feet_s;
}

py::array_t<double> compute_box_pos_s(const py::array_t<double>& box,
                             const py::array_t<double>& homo_inv) {
    auto box_buf = box.unchecked<1>();  // Assuming 2D array
    auto homo_inv_buf = homo_inv.unchecked<2>();  // Assuming 2D array

    double box_pos[2] = {(box_buf(0)+box_buf(2))/2, box_buf(3)};

    py::array_t<double> box_pos_w({2});
    auto box_pos_w_buf = box_pos_w.mutable_unchecked<1>();
    double box_pos_w_h[3];
    for (int j = 0; j < 3; j++) {
        box_pos_w_h[j] = homo_inv_buf(j, 0) * box_pos[0] + homo_inv_buf(j, 1) * box_pos[1] + homo_inv_buf(j, 2);
    }
    box_pos_w_buf(0) = box_pos_w_h[0]/box_pos_w_h[2];
    box_pos_w_buf(1) = box_pos_w_h[1]/box_pos_w_h[2];

    return box_pos_w;

}

std::vector<double> loop_t(
    const py::array_t<double> &joints_t,
    const py::array_t<double> &joints_s,
    const py::array_t<double> &age_bbox,
    const py::array_t<double> &age_2D,
    const py::array_t<double> &feet_s,
    bool feet_valid_s,
    pybind11::size_t v,
    double thred_epi,
    double thred_homo,
    double keypoint_thrd,
    double age_2D_thr,
    const py::array_t<double> &sv_ray,
    const py::list &cameras
) {
    int num_keypoints = joints_t.shape(1);
    auto joints_t_buf = joints_t.unchecked<3>();
    auto joints_s_buf = joints_s.unchecked<2>();
    auto age_bbox_buf = age_bbox.unchecked<1>();
    auto age_2D_buf = age_2D.unchecked<2>();
    auto pos = py::cast<py::array_t<double>>(cameras[v].attr("pos"));
    auto pos_buf = pos.unchecked<1>();
    auto sv_ray_buf = sv_ray.unchecked<2>();
    auto feet_s_buf = feet_s.unchecked<1>();  // Assuming 1D array

    double aff_ss_sum = 0;
    double aff_ss_cnt = 1e-5;
    double aff_homo_ss_sum = 0;
    double aff_homo_ss_cnt = 1e-5;

    for (py::size_t vj=0; vj<cameras.size(); vj++){
        if (v == vj || age_bbox_buf(vj) >= 2)
            continue;
        auto cam_pos = py::cast<py::array_t<double>>(cameras[vj].attr("pos"));
        auto cam_project_inv = py::cast<py::array_t<double>>(cameras[vj].attr("project_inv"));

        // track_rays_sv = track.CalcTargetRays(vj)
        auto cam_project_inv_buf = cam_project_inv.unchecked<2>();  // Assuming 2D array
        auto cam_pos_buf = cam_pos.unchecked<1>();  // Assuming 1D array

        double aff = 0;
        double aff_norm = 1e-5;
        for (int i = 0; i < num_keypoints; i++) {
            // cam.project_inv[:, :2] @ self.keypoints_mv[v][:,:-1].T
            if (joints_t_buf(vj, i, 2) > keypoint_thrd
            && joints_s_buf(i, 2) > keypoint_thrd
            && age_2D_buf(vj, i) <= age_2D_thr) {
                double joints_rays[4];
                for (int j = 0; j < 4; j++) {
                    joints_rays[j] = cam_project_inv_buf(j, 0) * joints_t_buf(vj, i, 0)
                                + cam_project_inv_buf(j, 1) * joints_t_buf(vj, i, 1)
                                + cam_project_inv_buf(j, 2);
                }
                for (int j = 0; j < 3; j++) {
                    joints_rays[j] = joints_rays[j] / joints_rays[3] - cam_pos_buf[j];
                }
                double norm = sqrt(joints_rays[0] * joints_rays[0] + joints_rays[1] * joints_rays[1] + joints_rays[2] * joints_rays[2]) + 1e-5;
                joints_rays[0] /= norm;
                joints_rays[1] /= norm;
                joints_rays[2] /= norm;

                double cp[3];  // cross
                cp[0] = sv_ray_buf(i, 1) * joints_rays[2] - sv_ray_buf(i, 2) * joints_rays[1];
                cp[1] = sv_ray_buf(i, 2) * joints_rays[0] - sv_ray_buf(i, 0) * joints_rays[2];
                cp[2] = sv_ray_buf(i, 0) * joints_rays[1] - sv_ray_buf(i, 1) * joints_rays[0];
                double _norm = sqrt(cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2]);
                double dot_product = (pos_buf(0) - cam_pos_buf(0)) * cp[0] +
                                    (pos_buf(1) - cam_pos_buf(1)) * cp[1] +
                                    (pos_buf(2) - cam_pos_buf(2)) * cp[2];
                double dist = std::abs(dot_product) / (_norm + 1e-6);
                double w = std::exp(-age_2D_buf(vj, i));
                aff += (1 - dist / thred_epi) * w;
                aff_norm += w;
            }
        }
        if (aff != 0){
            aff_ss_sum += aff / aff_norm;
            aff_ss_cnt += 1;
        }

        bool feet_valid_t = joints_t_buf(vj, 15, 2) > keypoint_thrd && joints_t_buf(vj, 16, 2) > keypoint_thrd;
        if (feet_valid_s && feet_valid_t) {
            auto homo_feet_inv = py::cast<py::array_t<double>>(cameras[vj].attr("homo_feet_inv"));;
            auto homo_feet_inv_buf = homo_feet_inv.unchecked<2>();  // Assuming 2D array

            double feet_pos[2] = {
                (joints_t_buf(vj, 15, 0) + joints_t_buf(vj, 16, 0)) / 2,
                (joints_t_buf(vj, 15, 1) + joints_t_buf(vj, 16, 1)) / 2
            };

            double feet_t_[3];
            for (int j = 0; j < 3; j++) {
                feet_t_[j] = homo_feet_inv_buf(j, 0) * feet_pos[0]
                        + homo_feet_inv_buf(j, 1) * feet_pos[1]
                        + homo_feet_inv_buf(j, 2);
            }
            double f0 = feet_s_buf(0) - feet_t_[0] / feet_t_[2];
            double f1 = feet_s_buf(1) - feet_t_[1] / feet_t_[2];
            double norm = sqrt(f0 * f0 + f1 * f1);
            // std::cout << 1 - norm / thred_homo << " ";
            aff_homo_ss_sum += 1 - norm / thred_homo;
            aff_homo_ss_cnt += 1;
        }
    }
    // std::cout << std::endl;
    return {aff_ss_sum / aff_ss_cnt, aff_homo_ss_sum / aff_homo_ss_cnt};
}

std::vector<double> loop_t_homo(
    const py::array_t<double> &joints_t,
    const py::array_t<double> &joints_s,
    const py::array_t<double> &age_bbox,
    const py::array_t<double> &age_2D,
    const py::array_t<double> &feet_s,
    bool feet_valid_s,
    pybind11::size_t v,
    double thred_epi,
    double thred_homo,
    double keypoint_thrd,
    double age_2D_thr,
    const py::array_t<double> &sv_ray,
    const py::list &cameras,
    const py::array_t<double> &bbox_s,
    bool box_valid_s,
    const py::array_t<double> &bbox_mv_t
) {
    int num_keypoints = joints_t.shape(1);
    auto joints_t_buf = joints_t.unchecked<3>();
    auto joints_s_buf = joints_s.unchecked<2>();
    auto age_bbox_buf = age_bbox.unchecked<1>();
    auto age_2D_buf = age_2D.unchecked<2>();
    auto pos = py::cast<py::array_t<double>>(cameras[v].attr("pos"));
    auto pos_buf = pos.unchecked<1>();
    auto sv_ray_buf = sv_ray.unchecked<2>();
    auto feet_s_buf = feet_s.unchecked<1>();  // Assuming 1D array
    auto bbox_s_buf = bbox_s.unchecked<1>(); 
    auto bbox_mv_t_buf = bbox_mv_t.unchecked<2>();


    double aff_ss_sum = 0;
    double aff_ss_cnt = 1e-5;
    double aff_homo_ss_sum = 0;
    double aff_homo_ss_cnt = 1e-5;

    for (py::size_t vj=0; vj<cameras.size(); vj++){
        if (v == vj || age_bbox_buf(vj) >= 2)
            continue;
        auto cam_pos = py::cast<py::array_t<double>>(cameras[vj].attr("pos"));
        auto cam_project_inv = py::cast<py::array_t<double>>(cameras[vj].attr("project_inv"));

        // track_rays_sv = track.CalcTargetRays(vj)
        auto cam_project_inv_buf = cam_project_inv.unchecked<2>();  // Assuming 2D array
        auto cam_pos_buf = cam_pos.unchecked<1>();  // Assuming 1D array

        double aff = 0;
        double aff_norm = 1e-5;
        for (int i = 0; i < num_keypoints; i++) {
            // cam.project_inv[:, :2] @ self.keypoints_mv[v][:,:-1].T
            if (joints_t_buf(vj, i, 2) > keypoint_thrd
            && joints_s_buf(i, 2) > keypoint_thrd
            && age_2D_buf(vj, i) <= age_2D_thr) {
                double joints_rays[4];
                for (int j = 0; j < 4; j++) {
                    joints_rays[j] = cam_project_inv_buf(j, 0) * joints_t_buf(vj, i, 0)
                                + cam_project_inv_buf(j, 1) * joints_t_buf(vj, i, 1)
                                + cam_project_inv_buf(j, 2);
                }
                for (int j = 0; j < 3; j++) {
                    joints_rays[j] = joints_rays[j] / joints_rays[3] - cam_pos_buf[j];
                }
                double norm = sqrt(joints_rays[0] * joints_rays[0] + joints_rays[1] * joints_rays[1] + joints_rays[2] * joints_rays[2]) + 1e-5;
                joints_rays[0] /= norm;
                joints_rays[1] /= norm;
                joints_rays[2] /= norm;

                double cp[3];  // cross
                cp[0] = sv_ray_buf(i, 1) * joints_rays[2] - sv_ray_buf(i, 2) * joints_rays[1];
                cp[1] = sv_ray_buf(i, 2) * joints_rays[0] - sv_ray_buf(i, 0) * joints_rays[2];
                cp[2] = sv_ray_buf(i, 0) * joints_rays[1] - sv_ray_buf(i, 1) * joints_rays[0];
                double _norm = sqrt(cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2]);
                double dot_product = (pos_buf(0) - cam_pos_buf(0)) * cp[0] +
                                    (pos_buf(1) - cam_pos_buf(1)) * cp[1] +
                                    (pos_buf(2) - cam_pos_buf(2)) * cp[2];
                double dist = std::abs(dot_product) / (_norm + 1e-6);
                double w = std::exp(-age_2D_buf(vj, i));
                aff += (1 - dist / thred_epi) * w;
                aff_norm += w;
            }
        }
        if (aff != 0){
            aff_ss_sum += aff / aff_norm;
            aff_ss_cnt += 1;
        }

        bool feet_valid_t = joints_t_buf(vj, 15, 2) > keypoint_thrd && joints_t_buf(vj, 16, 2) > keypoint_thrd;
        if (feet_valid_s && feet_valid_t) {
            auto homo_feet_inv = py::cast<py::array_t<double>>(cameras[vj].attr("homo_feet_inv"));;
            auto homo_feet_inv_buf = homo_feet_inv.unchecked<2>();  // Assuming 2D array

            double feet_pos[2] = {
                (joints_t_buf(vj, 15, 0) + joints_t_buf(vj, 16, 0)) / 2,
                (joints_t_buf(vj, 15, 1) + joints_t_buf(vj, 16, 1)) / 2
            };

            double feet_t_[3];
            for (int j = 0; j < 3; j++) {
                feet_t_[j] = homo_feet_inv_buf(j, 0) * feet_pos[0]
                        + homo_feet_inv_buf(j, 1) * feet_pos[1]
                        + homo_feet_inv_buf(j, 2);
            }
            double f0 = feet_s_buf(0) - feet_t_[0] / feet_t_[2];
            double f1 = feet_s_buf(1) - feet_t_[1] / feet_t_[2];
            double norm = sqrt(f0 * f0 + f1 * f1);
            // std::cout << 1 - norm / thred_homo << " ";
            aff_homo_ss_sum += 1 - norm / thred_homo;
            aff_homo_ss_cnt += 1;
        }
        else
        {
            auto homo_inv_vj = py::cast<py::array_t<double>>(cameras[vj].attr("homo_inv"));
            auto homo_inv_vj_buf = homo_inv_vj.unchecked<2>();

            if (bbox_mv_t_buf(vj,3)>=1075 || !box_valid_s)
            {
                // std::cout<<"bbox_mv_t_buf"<<bbox_mv_t_buf(vj,3)<<std::endl;
                // std::cout<<"box_valid_s"<<box_valid_s<<std::endl;
                // std::cout<<"sv "<<v<<" tv "<<vj<<std::endl;
                continue;
            }
               

            double box_pos[2] = {(bbox_mv_t_buf(vj,0)+bbox_mv_t_buf(vj,2))/2, bbox_mv_t_buf(vj,3)};
            
            double box_pos_w_h[3];
            for (int j = 0; j < 3; j++) {
                box_pos_w_h[j] = homo_inv_vj_buf(j, 0) * box_pos[0] + homo_inv_vj_buf(j, 1) * box_pos[1] + homo_inv_vj_buf(j, 2);
            }
            box_pos_w_h[0] = box_pos_w_h[0]/box_pos_w_h[2];
            box_pos_w_h[1] = box_pos_w_h[1]/box_pos_w_h[2];

            double f0 = box_pos_w_h[0] - bbox_s_buf(0);
            double f1 = box_pos_w_h[1] - bbox_s_buf(1);

            double norm = sqrt(f0 * f0 + f1 * f1);
            aff_homo_ss_sum += 1 - norm / thred_homo;
            aff_homo_ss_cnt += 1;

        }
    }
    // std::cout << std::endl;
    return {aff_ss_sum / aff_ss_cnt, aff_homo_ss_sum / aff_homo_ss_cnt};
}

std::vector<double> loop_t_homo_full(
    const py::array_t<double> &joints_t,
    const py::array_t<double> &joints_s,
    const py::array_t<double> &age_bbox,
    const py::array_t<double> &age_2D,
    const py::array_t<double> &feet_s,
    bool feet_valid_s,
    pybind11::size_t v,
    double thred_epi,
    double thred_homo,
    double keypoint_thrd,
    double age_2D_thr,
    const py::array_t<double> &sv_ray,
    const py::list &cameras,
    const py::array_t<double> &bbox_s,
    bool box_valid_s,
    const py::array_t<double> &bbox_mv_t
) {
    int num_keypoints = joints_t.shape(1);
    auto joints_t_buf = joints_t.unchecked<3>();
    auto joints_s_buf = joints_s.unchecked<2>();
    auto age_bbox_buf = age_bbox.unchecked<1>();
    auto age_2D_buf = age_2D.unchecked<2>();
    auto pos = py::cast<py::array_t<double>>(cameras[v].attr("pos"));
    auto pos_buf = pos.unchecked<1>();
    auto sv_ray_buf = sv_ray.unchecked<2>();
    auto feet_s_buf = feet_s.unchecked<1>();  // Assuming 1D array
    auto bbox_s_buf = bbox_s.unchecked<1>(); 
    auto bbox_mv_t_buf = bbox_mv_t.unchecked<2>();


    double aff_ss_sum = 0;
    double aff_ss_cnt = 1e-5;
    double aff_homo_ss_sum = 0;
    double aff_homo_ss_cnt = 1e-5;

    for (py::size_t vj=0; vj<cameras.size(); vj++){
        if (v == vj || age_bbox_buf(vj) >= 2)
            continue;
        auto cam_pos = py::cast<py::array_t<double>>(cameras[vj].attr("pos"));
        auto cam_project_inv = py::cast<py::array_t<double>>(cameras[vj].attr("project_inv"));

        // track_rays_sv = track.CalcTargetRays(vj)
        auto cam_project_inv_buf = cam_project_inv.unchecked<2>();  // Assuming 2D array
        auto cam_pos_buf = cam_pos.unchecked<1>();  // Assuming 1D array

        double aff = 0;
        double aff_norm = 1e-5;
        for (int i = 0; i < num_keypoints; i++) {
            // cam.project_inv[:, :2] @ self.keypoints_mv[v][:,:-1].T
            if (joints_t_buf(vj, i, 2) > keypoint_thrd
            && joints_s_buf(i, 2) > keypoint_thrd
            && age_2D_buf(vj, i) <= age_2D_thr) {
                double joints_rays[4];
                for (int j = 0; j < 4; j++) {
                    joints_rays[j] = cam_project_inv_buf(j, 0) * joints_t_buf(vj, i, 0)
                                + cam_project_inv_buf(j, 1) * joints_t_buf(vj, i, 1)
                                + cam_project_inv_buf(j, 2);
                }
                for (int j = 0; j < 3; j++) {
                    joints_rays[j] = joints_rays[j] / joints_rays[3] - cam_pos_buf[j];
                }
                double norm = sqrt(joints_rays[0] * joints_rays[0] + joints_rays[1] * joints_rays[1] + joints_rays[2] * joints_rays[2]) + 1e-5;
                joints_rays[0] /= norm;
                joints_rays[1] /= norm;
                joints_rays[2] /= norm;

                double cp[3];  // cross
                cp[0] = sv_ray_buf(i, 1) * joints_rays[2] - sv_ray_buf(i, 2) * joints_rays[1];
                cp[1] = sv_ray_buf(i, 2) * joints_rays[0] - sv_ray_buf(i, 0) * joints_rays[2];
                cp[2] = sv_ray_buf(i, 0) * joints_rays[1] - sv_ray_buf(i, 1) * joints_rays[0];
                double _norm = sqrt(cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2]);
                double dot_product = (pos_buf(0) - cam_pos_buf(0)) * cp[0] +
                                    (pos_buf(1) - cam_pos_buf(1)) * cp[1] +
                                    (pos_buf(2) - cam_pos_buf(2)) * cp[2];
                double dist = std::abs(dot_product) / (_norm + 1e-6);
                double w = std::exp(-age_2D_buf(vj, i));
                aff += (1 - dist / thred_epi) * w;
                aff_norm += w;
            }
        }
        if (aff != 0){
            aff_ss_sum += aff / aff_norm;
            aff_ss_cnt += 1;
        }

        bool feet_valid_t = joints_t_buf(vj, 15, 2) > keypoint_thrd && joints_t_buf(vj, 16, 2) > keypoint_thrd;
        if (feet_valid_s && feet_valid_t) {
            auto homo_feet_inv = py::cast<py::array_t<double>>(cameras[vj].attr("homo_feet_inv"));;
            auto homo_feet_inv_buf = homo_feet_inv.unchecked<2>();  // Assuming 2D array

            double feet_pos[2] = {
                (joints_t_buf(vj, 15, 0) + joints_t_buf(vj, 16, 0)) / 2,
                (joints_t_buf(vj, 15, 1) + joints_t_buf(vj, 16, 1)) / 2
            };

            double feet_t_[3];
            for (int j = 0; j < 3; j++) {
                feet_t_[j] = homo_feet_inv_buf(j, 0) * feet_pos[0]
                        + homo_feet_inv_buf(j, 1) * feet_pos[1]
                        + homo_feet_inv_buf(j, 2);
            }
            double f0 = feet_s_buf(0) - feet_t_[0] / feet_t_[2];
            double f1 = feet_s_buf(1) - feet_t_[1] / feet_t_[2];
            double norm = sqrt(f0 * f0 + f1 * f1);
            // std::cout << 1 - norm / thred_homo << " ";
            aff_homo_ss_sum += 1 - norm / thred_homo;
            aff_homo_ss_cnt += 1;
        }
        else
        {
            auto homo_inv_vj = py::cast<py::array_t<double>>(cameras[vj].attr("homo_inv"));
            auto homo_inv_vj_buf = homo_inv_vj.unchecked<2>();

            double box_pos[2] = {(bbox_mv_t_buf(vj,0)+bbox_mv_t_buf(vj,2))/2, bbox_mv_t_buf(vj,3)};
            
            double box_pos_w_h[3];
            for (int j = 0; j < 3; j++) {
                box_pos_w_h[j] = homo_inv_vj_buf(j, 0) * box_pos[0] + homo_inv_vj_buf(j, 1) * box_pos[1] + homo_inv_vj_buf(j, 2);
            }
            box_pos_w_h[0] = box_pos_w_h[0]/box_pos_w_h[2];
            box_pos_w_h[1] = box_pos_w_h[1]/box_pos_w_h[2];

            double f0 = box_pos_w_h[0] - bbox_s_buf(0);
            double f1 = box_pos_w_h[1] - bbox_s_buf(1);

            double norm = sqrt(f0 * f0 + f1 * f1);
            aff_homo_ss_sum += 1 - norm / thred_homo;
            aff_homo_ss_cnt += 1;

        }
    }
    // std::cout << std::endl;
    return {aff_ss_sum / aff_ss_cnt, aff_homo_ss_sum / aff_homo_ss_cnt};
}

py::array_t<double> bbox_overlap_rate(const py::array_t<double>& bboxes_s,
                                        const py::array_t<double>& bboxes_t) {

    int num_s = bboxes_s.shape(0);
    int num_t = bboxes_t.shape(0);

    auto bboxes_s_buf = bboxes_s.unchecked<2>();
    auto bboxes_t_buf = bboxes_t.unchecked<2>();

    py::array_t<double> overlap_rate({num_s, num_t});
    auto overlap_rate_buf = overlap_rate.mutable_unchecked<2>();

    for (int i=0;i<num_s;i++)
    {   
        double bbox_s[4] = {bboxes_s_buf(i,0),bboxes_s_buf(i,1),bboxes_s_buf(i,2),bboxes_s_buf(i,3)};
        double area_s = (bbox_s[2]-bbox_s[0]) * (bbox_s[3]-bbox_s[1]);
        for (int j=0;j<num_t;j++)
        {
            double bbox_t[4] = {bboxes_t_buf(j,0),bboxes_t_buf(j,1),bboxes_t_buf(j,2),bboxes_t_buf(j,3)};
            double x_left = std::max(bbox_s[0],bbox_t[0]);
            double y_top = std::max(bbox_s[1],bbox_t[1]);
            double x_right = std::min(bbox_s[2],bbox_t[2]);
            double y_bottom = std::min(bbox_s[3],bbox_t[3]);

            if (x_right<x_left || y_bottom < y_top)
            {
                overlap_rate_buf(i,j)=0;
            }
            else{
                double interset = (x_right - x_left) * (y_bottom - y_top);
                overlap_rate_buf(i,j) = interset/area_s;
            }

        }
    }

    return overlap_rate;                                     
                                        
}

PYBIND11_MODULE(aic_cpp, m) {
    m.def("epipolar_3d_score_norm", &epipolar_3d_score_norm, "");
    m.def("compute_joints_rays", &compute_joints_rays, "");
    m.def("aff_sum", &aff_sum, "");
    m.def("compute_feet_distance", &compute_feet_distance, "");
    m.def("compute_feet_s", &compute_feet_s, "");
    m.def("loop_t", &loop_t, "");
    m.def("bbox_overlap_rate", &bbox_overlap_rate, "");
    m.def("compute_box_pos_s", &compute_box_pos_s, "");
    m.def("loop_t_homo", &loop_t_homo, "");
    m.def("loop_t_homo_full", &loop_t_homo_full, "");
    
}
