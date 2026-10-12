#include "g1_pjsol_adaptor.hpp"

#include <spdlog/spdlog.h>

#include <cmath>

namespace {

constexpr int kMotorCount = 35;

struct AnkleSpec {
    const char* joint;
    const char* name_a;
    const char* name_b;
    int channel_a;
    int channel_b;
    int roll;
    int pitch;
};

// 名字到通道的映射与 config.yaml 注释一致。
constexpr AnkleSpec kAnkles[] = {
    {"left_ankle", "left_ankle_A", "left_ankle_B", 5, 4, 5, 4},
    {"right_ankle", "right_ankle_A", "right_ankle_B", 11, 10, 11, 10},
};

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool parseMode(const std::string& text, AbCommandMode* mode, std::string* error) {
    if (text == "torque") {
        *mode = AbCommandMode::Torque;
        return true;
    }
    if (text == "position") {
        *mode = AbCommandMode::Position;
        return true;
    }
    if (text == "hybrid") {
        *mode = AbCommandMode::Hybrid;
        return true;
    }
    return fail(error, "pjsol.command 必须是 torque、position 或 hybrid，当前是 " + text);
}

bool modeUsesAbGains(AbCommandMode mode) {
    return mode == AbCommandMode::Position || mode == AbCommandMode::Hybrid;
}

const char* modeName(AbCommandMode mode) {
    switch (mode) {
    case AbCommandMode::Torque:
        return "torque";
    case AbCommandMode::Position:
        return "position";
    case AbCommandMode::Hybrid:
        return "hybrid";
    }
    return "torque";
}

bool readVec3(const YAML::Node& node, const char* key, pjsol::Vec3& out, std::string* error) {
    const YAML::Node value = node[key];
    if (!value || !value.IsSequence() || value.size() != 3) {
        return fail(error, std::string(key) + " 需要 3 个数");
    }
    double xyz[3];
    for (int i = 0; i < 3; ++i) {
        xyz[i] = value[i].as<double>();
        if (!std::isfinite(xyz[i])) {
            return fail(error, std::string(key) + " 含非有限值");
        }
    }
    out = {xyz[0], xyz[1], xyz[2]};
    return true;
}

struct DriveAxis {
    int channel = 0;
    int direction = 1;
    double zero = 0.0;
};

bool readAxis(const YAML::Node& node, DriveAxis& axis, std::string* error) {
    if (!node || !node.IsMap()) {
        return fail(error, "drive_map 轴缺少字段");
    }
    axis.channel = node["channel"].as<int>();
    axis.direction = node["direction"].as<int>();
    const double zero_deg = node["zero_offset"].as<double>();
    if (axis.channel < 0 || axis.channel >= kMotorCount) {
        return fail(error, "drive_map.channel 超出电机数组");
    }
    if (axis.direction != 1 && axis.direction != -1) {
        return fail(error, "drive_map.direction 必须是 +1 或 -1");
    }
    if (!std::isfinite(zero_deg)) {
        return fail(error, "drive_map.zero_offset 不是有限值");
    }
    axis.zero = pjsol::math::deg2rad(zero_deg);
    return true;
}

bool sameChannels(int c1, int c2, int a, int b) {
    return (c1 == a && c2 == b) || (c1 == b && c2 == a);
}

void writePose(unitree_hg::msg::dds_::LowState_& state, int roll_channel, int pitch_channel,
               double roll, double pitch, double roll_dot, double pitch_dot) {
    state.motor_state()[roll_channel].q() = static_cast<float>(roll);
    state.motor_state()[pitch_channel].q() = static_cast<float>(pitch);
    state.motor_state()[roll_channel].dq() = static_cast<float>(roll_dot);
    state.motor_state()[pitch_channel].dq() = static_cast<float>(pitch_dot);
}

}  // namespace

G1PjsolAdaptor::G1PjsolAdaptor() = default;
G1PjsolAdaptor::~G1PjsolAdaptor() = default;

bool G1PjsolAdaptor::load(const YAML::Node& config, const std::filesystem::path& proj_dir,
                          std::string* error) {
    enabled_ = false;
    joints_.clear();
    state_modes_.clear();
    ab_kp_.clear();
    ab_kd_.clear();
    default_mode_ = AbCommandMode::Torque;

    const YAML::Node pjsol_node = config["pjsol"];
    if (!pjsol_node || pjsol_node.IsNull()) {
        return true;
    }
    if (!pjsol_node.IsMap()) {
        return fail(error, "pjsol 不是映射");
    }

    try {
        enabled_ = pjsol_node["enable"] ? pjsol_node["enable"].as<bool>() : false;
        if (!enabled_) {
            return true;
        }

        std::string mode_text = "torque";
        if (pjsol_node["command"]) {
            mode_text = pjsol_node["command"].as<std::string>();
        }
        if (!parseMode(mode_text, &default_mode_, error)) {
            return false;
        }

        bool needs_ab = modeUsesAbGains(default_mode_);
        const YAML::Node fsm = config["FSM"];
        if (fsm && fsm.IsMap()) {
            for (auto it = fsm.begin(); it != fsm.end(); ++it) {
                const std::string state_name = it->first.as<std::string>();
                if (state_name == "_" || !it->second["pjsol_command"]) {
                    continue;
                }
                AbCommandMode state_mode = AbCommandMode::Torque;
                const std::string state_text = it->second["pjsol_command"].as<std::string>();
                if (!parseMode(state_text, &state_mode, error)) {
                    if (error && !error->empty()) {
                        *error = state_name + ": " + *error;
                    }
                    return false;
                }
                state_modes_[state_name] = state_mode;
                needs_ab = needs_ab || modeUsesAbGains(state_mode);
            }
        }

        if (needs_ab) {
            const YAML::Node ab = pjsol_node["ab"];
            if (!ab || !ab.IsMap() || !ab["kp"] || !ab["kd"]) {
                return fail(error, "position 或 hybrid 需要 pjsol.ab.kp 和 pjsol.ab.kd");
            }
            auto readGains = [&](const YAML::Node& node, std::unordered_map<int, double>& out) -> bool {
                if (!node.IsMap()) {
                    return fail(error, "pjsol.ab 增益不是映射");
                }
                std::unordered_map<std::string, double> by_name;
                for (auto it = node.begin(); it != node.end(); ++it) {
                    const std::string name = it->first.as<std::string>();
                    const double value = it->second.as<double>();
                    if (!std::isfinite(value)) {
                        return fail(error, name + " 增益不是有限值");
                    }
                    by_name[name] = value;
                }
                for (const AnkleSpec& spec : kAnkles) {
                    if (!by_name.count(spec.name_a) || !by_name.count(spec.name_b)) {
                        return fail(error, std::string(spec.joint) + " 缺少 " + spec.name_a + " 或 " +
                                               spec.name_b + " 增益");
                    }
                    out[spec.channel_a] = by_name[spec.name_a];
                    out[spec.channel_b] = by_name[spec.name_b];
                    by_name.erase(spec.name_a);
                    by_name.erase(spec.name_b);
                }
                if (!by_name.empty()) {
                    return fail(error, "pjsol.ab 含未知电机 " + by_name.begin()->first);
                }
                return true;
            };
            if (!readGains(ab["kp"], ab_kp_) || !readGains(ab["kd"], ab_kd_)) {
                return false;
            }
        }

        if (!pjsol_node["param_file"]) {
            return fail(error, "pjsol.param_file 缺失");
        }
        std::filesystem::path param_path = pjsol_node["param_file"].as<std::string>();
        if (param_path.is_relative()) {
            param_path = proj_dir / param_path;
        }
        YAML::Node param_root;
        try {
            param_root = YAML::LoadFile(param_path.string());
        } catch (const YAML::Exception& ex) {
            return fail(error, "读取 " + param_path.string() + " 失败: " + ex.what());
        }
        if (!param_root || !param_root.IsMap()) {
            return fail(error, param_path.string() + " 不是映射");
        }

        std::unordered_map<std::string, bool> seen;
        for (const AnkleSpec& spec : kAnkles) {
            seen[spec.joint] = false;
        }
        for (auto it = param_root.begin(); it != param_root.end(); ++it) {
            const std::string name = it->first.as<std::string>();
            if (!seen.count(name)) {
                return fail(error, param_path.string() + " 含未知关节 " + name);
            }
            seen[name] = true;

            const AnkleSpec* spec = nullptr;
            for (const AnkleSpec& item : kAnkles) {
                if (name == item.joint) {
                    spec = &item;
                    break;
                }
            }
            const YAML::Node node = it->second;
            if (!node || !node.IsMap()) {
                return fail(error, name + " 不是映射");
            }

            Joint joint;
            joint.name = name;
            const std::string type = node["type"].as<std::string>();
            if (type.empty()) {
                return fail(error, name + ".type 为空");
            }
            DriveAxis q1{};
            DriveAxis q2{};
            std::string axis_error;
            if (!readAxis(node["drive_map"]["q1"], q1, &axis_error) ||
                !readAxis(node["drive_map"]["q2"], q2, &axis_error)) {
                return fail(error, name + ": " + axis_error);
            }
            joint.q1 = {q1.channel, q1.direction, q1.zero};
            joint.q2 = {q2.channel, q2.direction, q2.zero};
            if (q1.channel == q2.channel) {
                return fail(error, name + ": q1 与 q2 的 channel 相同");
            }
            if (!sameChannels(q1.channel, q2.channel, spec->channel_a, spec->channel_b)) {
                return fail(error, name + ": drive_map.channel 与 " + std::string(spec->name_a) + "/" +
                                       spec->name_b + " 不一致");
            }

            joint.roll_channel = node["output_map"]["roll"].as<int>();
            joint.pitch_channel = node["output_map"]["pitch"].as<int>();
            if (joint.roll_channel != spec->roll || joint.pitch_channel != spec->pitch) {
                return fail(error, name + ": output_map 与 PR 通道不一致");
            }

            const YAML::Node geom_node = node["geom"];
            pjsol::Geom geom{};
            std::string geom_error;
            if (!readVec3(geom_node, "F1", geom.F1, &geom_error) ||
                !readVec3(geom_node, "F2", geom.F2, &geom_error) ||
                !readVec3(geom_node, "H1", geom.H1, &geom_error) ||
                !readVec3(geom_node, "H2", geom.H2, &geom_error)) {
                return fail(error, name + ": " + geom_error);
            }
            geom.l1 = geom_node["l1"].as<double>();
            geom.l2 = geom_node["l2"].as<double>();
            geom.r = geom_node["r"].as<double>();
            geom.e = geom_node["e"].as<double>();
            if (!std::isfinite(geom.l1) || !std::isfinite(geom.l2) || !std::isfinite(geom.r) ||
                !std::isfinite(geom.e) || geom.l1 <= 0.0 || geom.l2 <= 0.0 || geom.r <= 0.0) {
                return fail(error, name + ": geom 的 l1/l2/r/e 无效");
            }

            const YAML::Node limit_node = node["joint_limit"];
            const double q1_min = limit_node["q1_min"].as<double>();
            const double q1_max = limit_node["q1_max"].as<double>();
            const double q2_min = limit_node["q2_min"].as<double>();
            const double q2_max = limit_node["q2_max"].as<double>();
            const double sphere = limit_node["sphere_angle_max"].as<double>();
            if (!std::isfinite(q1_min) || !std::isfinite(q1_max) || !std::isfinite(q2_min) ||
                !std::isfinite(q2_max) || !std::isfinite(sphere) || !(q1_min < q1_max) ||
                !(q2_min < q2_max) || sphere <= 0.0 || sphere > 90.0) {
                return fail(error, name + ": joint_limit 无效（单位是度）");
            }

            pjsol::JointLimit limit{};
            limit.active.q1_min = pjsol::math::deg2rad(q1_min);
            limit.active.q1_max = pjsol::math::deg2rad(q1_max);
            limit.active.q2_min = pjsol::math::deg2rad(q2_min);
            limit.active.q2_max = pjsol::math::deg2rad(q2_max);
            limit.passive.sphere_angle_max = pjsol::math::deg2rad(sphere);

            joint.mech = pjsol::create(type, geom, limit);
            if (!joint.mech) {
                return fail(error, name + ": pjsol::create 失败，type=" + type);
            }
            joints_.push_back(std::move(joint));
        }
        for (const auto& item : seen) {
            if (!item.second) {
                return fail(error, std::string(param_path.string()) + " 缺少 " + item.first);
            }
        }
    } catch (const YAML::Exception& ex) {
        return fail(error, std::string("pjsol 配置错误: ") + ex.what());
    }

    spdlog::info("G1 pjsol adaptor enabled, command={}, joints={}", modeName(default_mode_),
                 joints_.size());
    return true;
}

pjsol::Joint2 G1PjsolAdaptor::modelPosition(const Joint& joint) const {
    return {joint.q1.direction * (joint.ab_q1 - joint.q1.zero),
            joint.q2.direction * (joint.ab_q2 - joint.q2.zero)};
}

pjsol::JointVel2 G1PjsolAdaptor::modelVelocity(const Joint& joint) const {
    return {joint.q1.direction * joint.ab_dq1, joint.q2.direction * joint.ab_dq2};
}

AbCommandMode G1PjsolAdaptor::modeFor(const std::string& fsm_state) const {
    const auto it = state_modes_.find(fsm_state);
    if (it == state_modes_.end()) {
        return default_mode_;
    }
    return it->second;
}

void G1PjsolAdaptor::writePreviousPose(const Joint& joint,
                                       unitree_hg::msg::dds_::LowState_& state) const {
    if (joint.pose_valid) {
        writePose(state, joint.roll_channel, joint.pitch_channel, joint.pose.qr, joint.pose.qp,
                  joint.vel.dqr, joint.vel.dqp);
        return;
    }
    writePose(state, joint.roll_channel, joint.pitch_channel, 0.0, 0.0, 0.0, 0.0);
}

bool G1PjsolAdaptor::updateJointFeedback(Joint& joint, unitree_hg::msg::dds_::LowState_& state) {
    joint.ab_q1 = state.motor_state()[joint.q1.channel].q();
    joint.ab_q2 = state.motor_state()[joint.q2.channel].q();
    joint.ab_dq1 = state.motor_state()[joint.q1.channel].dq();
    joint.ab_dq2 = state.motor_state()[joint.q2.channel].dq();

    const pjsol::Joint2 q = modelPosition(joint);
    const pjsol::JointVel2 dq = modelVelocity(joint);
    pjsol::Pose2 x{};
    if (!joint.mech->fk(q, x, joint.pose)) {
        writePreviousPose(joint, state);
        return false;
    }

    pjsol::Mat2 Jx2q{};
    pjsol::Mat2 Jq2x{};
    if (!joint.mech->jacobian(x, q, Jx2q, Jq2x)) {
        writePreviousPose(joint, state);
        return false;
    }

    joint.pose = x;
    joint.vel = joint.mech->vel_q2x(Jq2x, dq);
    joint.pose_valid = true;
    writePose(state, joint.roll_channel, joint.pitch_channel, joint.pose.qr, joint.pose.qp,
              joint.vel.dqr, joint.vel.dqp);
    return true;
}

void G1PjsolAdaptor::updatePrState(unitree_hg::msg::dds_::LowState_& state) {
    if (!enabled_) {
        return;
    }
    bool fault = false;
    for (Joint& joint : joints_) {
        if (!updateJointFeedback(joint, state)) {
            fault = true;
            if (!fault_logged_) {
                spdlog::error("{} 正解失败，切换 Passive", joint.name);
            }
        }
    }
    if (fault && !fault_logged_) {
        fault_logged_ = true;
    }
    if (!fault) {
        fault_logged_ = false;
    }
    feedback_fault_ = fault;
}

void G1PjsolAdaptor::zeroDrive(unitree_hg::msg::dds_::LowCmd_& cmd, const Joint& joint) const {
    for (const int channel : {joint.q1.channel, joint.q2.channel}) {
        auto& motor = cmd.motor_cmd()[channel];
        motor.q() = 0.0f;
        motor.dq() = 0.0f;
        motor.kp() = 0.0f;
        motor.kd() = 0.0f;
        motor.tau() = 0.0f;
    }
}

bool G1PjsolAdaptor::solveTorque(const Joint& joint, double roll_des, double pitch_des,
                                 double roll_dot_des, double pitch_dot_des, double kp_roll,
                                 double kp_pitch, double kd_roll, double kd_pitch, double* tau1,
                                 double* tau2) const {
    const pjsol::Joint2 q = modelPosition(joint);
    const pjsol::JointVel2 dq = modelVelocity(joint);
    pjsol::Pose2 x{};
    if (!joint.mech->fk(q, x, joint.pose)) {
        return false;
    }
    pjsol::Mat2 Jx2q{};
    pjsol::Mat2 Jq2x{};
    if (!joint.mech->jacobian(x, q, Jx2q, Jq2x)) {
        return false;
    }
    const pjsol::PoseVel2 dx = joint.mech->vel_q2x(Jq2x, dq);
    const pjsol::PoseTor2 tor_x{
        kp_roll * (roll_des - x.qr) + kd_roll * (roll_dot_des - dx.dqr),
        kp_pitch * (pitch_des - x.qp) + kd_pitch * (pitch_dot_des - dx.dqp)};
    const pjsol::JointTor2 tor_q = joint.mech->tor_x2q(Jq2x, tor_x);
    *tau1 = joint.q1.direction * tor_q.tau1;
    *tau2 = joint.q2.direction * tor_q.tau2;
    return true;
}

void G1PjsolAdaptor::writeTorque(Joint& joint, unitree_hg::msg::dds_::LowCmd_& cmd, double roll_des,
                                 double pitch_des, double roll_dot_des, double pitch_dot_des,
                                 double kp_roll, double kp_pitch, double kd_roll, double kd_pitch) {
    double tau1 = joint.tau1;
    double tau2 = joint.tau2;
    if (solveTorque(joint, roll_des, pitch_des, roll_dot_des, pitch_dot_des, kp_roll, kp_pitch,
                    kd_roll, kd_pitch, &tau1, &tau2)) {
        joint.tau1 = tau1;
        joint.tau2 = tau2;
        joint.torque_fault = false;
    } else if (!joint.torque_fault) {
        spdlog::error("{} 力矩映射失败，保持上一力矩", joint.name);
        joint.torque_fault = true;
    }
    zeroDrive(cmd, joint);
    cmd.motor_cmd()[joint.q1.channel].tau() = static_cast<float>(joint.tau1);
    cmd.motor_cmd()[joint.q2.channel].tau() = static_cast<float>(joint.tau2);
}

bool G1PjsolAdaptor::solvePosition(const Joint& joint, double roll_des, double pitch_des,
                                   double roll_dot_des, double pitch_dot_des, double* motor_q1,
                                   double* motor_q2, double* motor_dq1, double* motor_dq2) const {
    const pjsol::Pose2 x{roll_des, pitch_des};
    pjsol::Joint2 q{};
    if (!joint.mech->ik(x, q)) {
        return false;
    }
    pjsol::JointVel2 dq{};
    pjsol::Mat2 Jx2q{};
    pjsol::Mat2 Jq2x{};
    if (joint.mech->jacobian(x, q, Jx2q, Jq2x)) {
        dq = joint.mech->vel_x2q(Jx2q, pjsol::PoseVel2{roll_dot_des, pitch_dot_des});
    }
    *motor_q1 = joint.q1.direction * q.q1 + joint.q1.zero;
    *motor_q2 = joint.q2.direction * q.q2 + joint.q2.zero;
    *motor_dq1 = joint.q1.direction * dq.dq1;
    *motor_dq2 = joint.q2.direction * dq.dq2;
    return true;
}

void G1PjsolAdaptor::writePosition(Joint& joint, unitree_hg::msg::dds_::LowCmd_& cmd,
                                   double roll_des, double pitch_des, double roll_dot_des,
                                   double pitch_dot_des) {
    double motor_q1 = 0.0;
    double motor_q2 = 0.0;
    double motor_dq1 = 0.0;
    double motor_dq2 = 0.0;
    if (!solvePosition(joint, roll_des, pitch_des, roll_dot_des, pitch_dot_des, &motor_q1,
                       &motor_q2, &motor_dq1, &motor_dq2)) {
        if (!joint.position_fault) {
            spdlog::error("{} 逆解失败，AB 指令清零", joint.name);
            joint.position_fault = true;
        }
        zeroDrive(cmd, joint);
        return;
    }
    joint.position_fault = false;
    auto& m1 = cmd.motor_cmd()[joint.q1.channel];
    auto& m2 = cmd.motor_cmd()[joint.q2.channel];
    m1.q() = static_cast<float>(motor_q1);
    m2.q() = static_cast<float>(motor_q2);
    m1.dq() = static_cast<float>(motor_dq1);
    m2.dq() = static_cast<float>(motor_dq2);
    m1.kp() = static_cast<float>(ab_kp_.at(joint.q1.channel));
    m2.kp() = static_cast<float>(ab_kp_.at(joint.q2.channel));
    m1.kd() = static_cast<float>(ab_kd_.at(joint.q1.channel));
    m2.kd() = static_cast<float>(ab_kd_.at(joint.q2.channel));
    m1.tau() = 0.0f;
    m2.tau() = 0.0f;
}

void G1PjsolAdaptor::writeHybrid(Joint& joint, unitree_hg::msg::dds_::LowCmd_& cmd, double roll_des,
                                 double pitch_des, double roll_dot_des, double pitch_dot_des,
                                 double kp_roll, double kp_pitch, double kd_roll, double kd_pitch) {
    double motor_q1 = 0.0;
    double motor_q2 = 0.0;
    double motor_dq1 = 0.0;
    double motor_dq2 = 0.0;
    if (solvePosition(joint, roll_des, pitch_des, roll_dot_des, pitch_dot_des, &motor_q1, &motor_q2,
                      &motor_dq1, &motor_dq2)) {
        joint.position_fault = false;
        auto& m1 = cmd.motor_cmd()[joint.q1.channel];
        auto& m2 = cmd.motor_cmd()[joint.q2.channel];
        m1.q() = static_cast<float>(motor_q1);
        m2.q() = static_cast<float>(motor_q2);
        m1.dq() = static_cast<float>(motor_dq1);
        m2.dq() = static_cast<float>(motor_dq2);
        m1.kp() = static_cast<float>(ab_kp_.at(joint.q1.channel));
        m2.kp() = static_cast<float>(ab_kp_.at(joint.q2.channel));
        m1.kd() = static_cast<float>(ab_kd_.at(joint.q1.channel));
        m2.kd() = static_cast<float>(ab_kd_.at(joint.q2.channel));
        m1.tau() = 0.0f;
        m2.tau() = 0.0f;
        return;
    }
    if (!joint.position_fault) {
        spdlog::warn("{} 逆解失败，改用力矩映射", joint.name);
        joint.position_fault = true;
    }
    writeTorque(joint, cmd, roll_des, pitch_des, roll_dot_des, pitch_dot_des, kp_roll, kp_pitch,
                kd_roll, kd_pitch);
}

void G1PjsolAdaptor::updateAbCommand(unitree_hg::msg::dds_::LowCmd_& cmd,
                                     const std::string& fsm_state) {
    if (!enabled_) {
        return;
    }
    if (feedback_fault_) {
        for (const Joint& joint : joints_) {
            zeroDrive(cmd, joint);
        }
        return;
    }

    const AbCommandMode mode = modeFor(fsm_state);
    for (Joint& joint : joints_) {
        const double roll_des = cmd.motor_cmd()[joint.roll_channel].q();
        const double pitch_des = cmd.motor_cmd()[joint.pitch_channel].q();
        const double roll_dot_des = cmd.motor_cmd()[joint.roll_channel].dq();
        const double pitch_dot_des = cmd.motor_cmd()[joint.pitch_channel].dq();
        const double kp_roll = cmd.motor_cmd()[joint.roll_channel].kp();
        const double kp_pitch = cmd.motor_cmd()[joint.pitch_channel].kp();
        const double kd_roll = cmd.motor_cmd()[joint.roll_channel].kd();
        const double kd_pitch = cmd.motor_cmd()[joint.pitch_channel].kd();

        switch (mode) {
        case AbCommandMode::Torque:
            writeTorque(joint, cmd, roll_des, pitch_des, roll_dot_des, pitch_dot_des, kp_roll,
                        kp_pitch, kd_roll, kd_pitch);
            break;
        case AbCommandMode::Position:
            writePosition(joint, cmd, roll_des, pitch_des, roll_dot_des, pitch_dot_des);
            break;
        case AbCommandMode::Hybrid:
            writeHybrid(joint, cmd, roll_des, pitch_des, roll_dot_des, pitch_dot_des, kp_roll,
                        kp_pitch, kd_roll, kd_pitch);
            break;
        }
    }
}
