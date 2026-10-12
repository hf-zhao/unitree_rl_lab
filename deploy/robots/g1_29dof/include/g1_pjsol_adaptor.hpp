#pragma once

#include <pjsol/pjsol.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

enum class AbCommandMode { Torque, Position, Hybrid };

// G1 并联关节适配器。反馈把 AB 电机角解成 PR，指令按 torque / position / hybrid 写回 AB。
class G1PjsolAdaptor {
public:
    G1PjsolAdaptor();
    ~G1PjsolAdaptor();

    bool load(const YAML::Node& config, const std::filesystem::path& proj_dir, std::string* error);
    bool enabled() const { return enabled_; }

    // 本拍任一关节正解或速度雅可比失败。CtrlFSM 用它切到 Passive。
    bool feedbackFault() const { return feedback_fault_; }

    // 读并联关节 AB 的 q/dq，写成 PR 的 pitch/roll 和速度，写回 LowState。
    void updatePrState(unitree_hg::msg::dds_::LowState_& state);

    // 读本拍 PR 目标，按 fsm_state 的输出方式写成 AB，写回 LowCmd。
    void updateAbCommand(unitree_hg::msg::dds_::LowCmd_& cmd, const std::string& fsm_state);

private:
    struct Axis {
        int channel = 0;
        int direction = 1;
        double zero = 0.0;
    };

    struct Joint {
        std::string name;
        std::unique_ptr<pjsol::IMechanism> mech;
        Axis q1;
        Axis q2;
        int roll_channel = 0;
        int pitch_channel = 0;
        pjsol::Pose2 pose{};
        pjsol::PoseVel2 vel{};
        bool pose_valid = false;
        double ab_q1 = 0.0;
        double ab_q2 = 0.0;
        double ab_dq1 = 0.0;
        double ab_dq2 = 0.0;
        double tau1 = 0.0;
        double tau2 = 0.0;
        bool torque_fault = false;
        bool position_fault = false;
    };

    AbCommandMode modeFor(const std::string& fsm_state) const;
    pjsol::Joint2 modelPosition(const Joint& joint) const;
    pjsol::JointVel2 modelVelocity(const Joint& joint) const;
    bool updateJointFeedback(Joint& joint, unitree_hg::msg::dds_::LowState_& state);
    void writePreviousPose(const Joint& joint, unitree_hg::msg::dds_::LowState_& state) const;
    void zeroDrive(unitree_hg::msg::dds_::LowCmd_& cmd, const Joint& joint) const;
    bool solveTorque(const Joint& joint, double roll_des, double pitch_des, double roll_dot_des,
                     double pitch_dot_des, double kp_roll, double kp_pitch, double kd_roll,
                     double kd_pitch, double* tau1, double* tau2) const;
    void writeTorque(Joint& joint, unitree_hg::msg::dds_::LowCmd_& cmd, double roll_des,
                     double pitch_des, double roll_dot_des, double pitch_dot_des, double kp_roll,
                     double kp_pitch, double kd_roll, double kd_pitch);
    bool solvePosition(const Joint& joint, double roll_des, double pitch_des, double roll_dot_des,
                       double pitch_dot_des, double* motor_q1, double* motor_q2, double* motor_dq1,
                       double* motor_dq2) const;
    void writePosition(Joint& joint, unitree_hg::msg::dds_::LowCmd_& cmd, double roll_des,
                       double pitch_des, double roll_dot_des, double pitch_dot_des);
    void writeHybrid(Joint& joint, unitree_hg::msg::dds_::LowCmd_& cmd, double roll_des,
                     double pitch_des, double roll_dot_des, double pitch_dot_des, double kp_roll,
                     double kp_pitch, double kd_roll, double kd_pitch);

    bool enabled_ = false;
    bool feedback_fault_ = false;
    bool fault_logged_ = false;
    AbCommandMode default_mode_ = AbCommandMode::Torque;
    std::unordered_map<std::string, AbCommandMode> state_modes_;
    std::unordered_map<int, double> ab_kp_;
    std::unordered_map<int, double> ab_kd_;
    std::vector<Joint> joints_;
};
