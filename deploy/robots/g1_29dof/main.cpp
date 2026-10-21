#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "g1_pjsol_adaptor.hpp"
#include "joint_recorder.h"

#include <atomic>
#include <csignal>
#include <memory>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();
static std::shared_ptr<G1PjsolAdaptor> g1_pjsol_adaptor;

static std::atomic<bool> g_running{true};

static void on_signal(int)
{
    // 第二次 Ctrl+C 强制退出，防止 shutdown 卡住
    if (!g_running.exchange(false)) {
        std::_Exit(130);
    }
}

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";
    if (param::config["mode_machine"] && param::config["mode_machine"].as<int>() == 6) {
        std::cout << "     (lock waist) \n";
    }

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // 5: g1_29dof_rev_1_0, 6: g1_29dof_lock_waist_rev_1_0
    const int mode_machine = param::config["mode_machine"] ? param::config["mode_machine"].as<int>() : 5;
    FSMState::lowcmd->msg_.mode_machine() = mode_machine;
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type. config mode_machine={}, robot mode_machine={}",
            mode_machine, FSMState::lowstate->msg_.mode_machine());
        exit(-1);
    }

    g1_pjsol_adaptor = std::make_shared<G1PjsolAdaptor>();
    std::string pjsol_error;
    if (!g1_pjsol_adaptor->load(param::config, param::proj_dir, &pjsol_error)) {
        spdlog::critical("{}", pjsol_error);
        exit(-1);
    }
    if (g1_pjsol_adaptor->enabled()) {
        FSMState::lowcmd->msg_.mode_pr() = 1;
        FSMState::pjsol_update_pr_state = [] {
            g1_pjsol_adaptor->updatePrState(FSMState::lowstate->msg_);
        };
        FSMState::pjsol_update_ab_command = [](const std::string& fsm_state) {
            g1_pjsol_adaptor->updateAbCommand(FSMState::lowcmd->msg_, fsm_state);
        };
        FSMState::pjsol_feedback_fault = [] {
            return g1_pjsol_adaptor->feedbackFault();
        };
    }
    
    JointRecorder::instance().start();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";

    while (g_running)
    {
        sleep(1);
    }

    fsm.reset();
    JointRecorder::instance().shutdown();
    
    return 0;
}
