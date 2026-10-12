#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

#include <functional>
#include <string>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
        // AB 模式下正解失败切到 Passive。未启用时回调为空，检查为假。
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool {
                    if (!pjsol_feedback_fault) return false;
                    return pjsol_feedback_fault();
                },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        // 启用 AB 模式时，把电机反馈解成等效 PR，供本拍控制使用。
        if (pjsol_update_pr_state) pjsol_update_pr_state();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        // 启用 AB 模式时，把本拍 PR 指令写成实际 AB 电机指令后再发布。
        if (pjsol_update_ab_command) pjsol_update_ab_command(getStateString());
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
    // 仅在 AB 模式启用时赋值。空回调表示继续使用板载 PR。
    inline static std::function<void()> pjsol_update_pr_state;
    inline static std::function<void(const std::string&)> pjsol_update_ab_command;
    inline static std::function<bool()> pjsol_feedback_fault;
};