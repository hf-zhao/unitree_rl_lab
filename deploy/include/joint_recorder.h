// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include "param.h"

/**
 * 记录策略期望关节位置与实测关节状态，并在期望位置超出关节限位时发出警告。
 *
 * sample() 在控制线程中调用：只做数据拷贝，写入预分配缓冲。
 * 文件 I/O 与控制台警告由后台线程处理（约每 100ms 批量写盘，越限警告每 1s 汇总一次）。
 *
 * 各关节数据均为 SDK 电机顺序。关节名与限位来自 config.yaml 的
 * `joint_names` / `joint_pos_limits`。
 */
class JointRecorder
{
public:
    static constexpr int kMaxJoints = 35;

    static JointRecorder& instance()
    {
        static JointRecorder recorder;
        return recorder;
    }

    void begin(const std::string& state_name)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        session_names_.push_back(timestamp() + "_" + state_name);
        session_ = session_names_.size() - 1;
        session_start_ = clock::now();
        active_ = true;
        if (!worker_.joinable()) {
            worker_ = std::thread(&JointRecorder::worker_loop, this);
        }
    }

    void end()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = false;
    }

    /**
     * @param joint_ids_map 策略关节下标 i -> SDK 电机下标
     * @param q_des 策略期望关节位置（策略顺序）
     */
    template <typename LowStatePtr>
    void sample(const LowStatePtr& lowstate, const std::vector<float>& joint_ids_map,
                const std::vector<float>& q_des, long policy_step)
    {
        Sample s;
        s.n = std::min<int>(names_.empty() ? joint_ids_map.size() : names_.size(), kMaxJoints);
        s.q_des.fill(std::numeric_limits<float>::quiet_NaN());
        for (size_t i = 0; i < joint_ids_map.size() && i < q_des.size(); ++i) {
            int m = static_cast<int>(joint_ids_map[i]);
            if (m >= 0 && m < s.n) s.q_des[m] = q_des[i];
        }
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            const auto& motors = lowstate->msg_.motor_state();
            s.n = std::min<int>(s.n, motors.size());
            for (int j = 0; j < s.n; ++j) {
                s.q[j] = motors[j].q();
                s.dq[j] = motors[j].dq();
                s.tau[j] = motors[j].tau_est();
            }
            s.tick = lowstate->msg_.tick();
        }
        s.step = policy_step;

        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_) return;
        s.session = session_;
        s.t = std::chrono::duration<double>(clock::now() - session_start_).count();

        for (int j = 0; j < s.n && j < static_cast<int>(limits_.size()); ++j) {
            const float q = s.q_des[j];
            const auto& [lo, hi] = limits_[j];
            if (std::isnan(q) || (q >= lo && q <= hi)) continue;
            const float exceed = q < lo ? lo - q : q - hi;
            auto& st = limit_stats_[j];
            if (st.count == 0 || exceed > st.worst_exceed) {
                st.worst_exceed = exceed;
                st.worst_q = q;
            }
            ++st.count;
        }

        if (!param::record) return;
        if (pending_.size() < kMaxPending) {
            pending_.push_back(s);
        } else {
            ++dropped_;
        }
    }

    ~JointRecorder()
    {
        stop_ = true;
        if (worker_.joinable()) worker_.join();
    }

private:
    using clock = std::chrono::steady_clock;
    static constexpr size_t kMaxPending = 4096;

    struct Sample {
        size_t session;
        double t;
        uint32_t tick;
        long step;
        int n;
        std::array<float, kMaxJoints> q_des, q, dq, tau;
    };

    struct LimitStat {
        int count = 0;
        float worst_q = 0.0f;
        float worst_exceed = 0.0f;
    };

    JointRecorder()
    {
        // 保证 spdlog 生命周期长于本单例：析构时 worker 仍可能打日志
        spdlog::default_logger();
        pending_.reserve(kMaxPending);

        if (auto names = param::config["joint_names"]) {
            names_ = names.as<std::vector<std::string>>();
        }
        if (auto limits = param::config["joint_pos_limits"]) {
            for (const auto& l : limits) {
                auto v = l.as<std::vector<float>>();
                limits_.emplace_back(v.at(0), v.at(1));
            }
        }
        if (limits_.empty()) {
            spdlog::warn("config.yaml 未配置 joint_pos_limits，已禁用策略期望位置限位检查。");
        } else if (!names_.empty() && names_.size() != limits_.size()) {
            spdlog::warn("joint_names ({}) 与 joint_pos_limits ({}) 数量不一致，已禁用限位检查。",
                         names_.size(), limits_.size());
            limits_.clear();
        }
    }

    std::string joint_name(int j) const
    {
        return j < static_cast<int>(names_.size()) ? names_[j] : fmt::format("joint_{}", j);
    }

    static std::string timestamp()
    {
        std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm{};
        localtime_r(&now, &tm);
        std::ostringstream ss;
        ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        return ss.str();
    }

    void worker_loop()
    {
        std::vector<Sample> batch;
        batch.reserve(kMaxPending);
        std::ofstream ofs;
        std::filesystem::path file_path;
        size_t file_session = std::numeric_limits<size_t>::max();
        fmt::memory_buffer buf;
        auto last_report = clock::now();

        auto close_file = [&]() {
            if (!ofs.is_open()) return;
            ofs.close();
            spdlog::info("关节数据已保存至 {}", file_path.string());
        };

        while (!stop_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            bool active;
            bool report = clock::now() - last_report >= std::chrono::seconds(1);
            std::array<LimitStat, kMaxJoints> stats;
            size_t dropped = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                batch.swap(pending_);
                active = active_;
                if (report) {
                    stats = limit_stats_;
                    limit_stats_ = {};
                    dropped = dropped_;
                    dropped_ = 0;
                }
            }

            for (const auto& s : batch) {
                if (s.session != file_session) {
                    close_file();
                    std::string stem;
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        stem = session_names_[s.session];
                    }
                    std::filesystem::create_directories(param::record_dir);
                    file_path = param::record_dir / (stem + ".csv");
                    ofs.open(file_path);
                    if (!ofs) spdlog::error("无法打开文件 {}", file_path.string());
                    file_session = s.session;

                    fmt::format_to(std::back_inserter(buf), "t,tick,step");
                    for (int j = 0; j < s.n; ++j) {
                        auto name = joint_name(j);
                        fmt::format_to(std::back_inserter(buf), ",{0}_q_des,{0}_q,{0}_dq,{0}_tau_est", name);
                    }
                    buf.push_back('\n');
                }
                fmt::format_to(std::back_inserter(buf), "{:.4f},{},{}", s.t, s.tick, s.step);
                for (int j = 0; j < s.n; ++j) {
                    fmt::format_to(std::back_inserter(buf), ",{:.6g},{:.6g},{:.6g},{:.6g}",
                                   s.q_des[j], s.q[j], s.dq[j], s.tau[j]);
                }
                buf.push_back('\n');
            }
            batch.clear();
            if (ofs.is_open() && buf.size() > 0) {
                ofs.write(buf.data(), buf.size());
                ofs.flush();
            }
            buf.clear();
            if (!active) close_file();

            if (report) {
                last_report = clock::now();
                constexpr float rad2deg = 180.0f / M_PI;
                for (int j = 0; j < kMaxJoints && j < static_cast<int>(limits_.size()); ++j) {
                    if (stats[j].count == 0) continue;
                    spdlog::warn("[{}] 策略输出的期望位置 {:.1f} deg，超过关节范围 [{:.1f}, {:.1f}] deg（近 1s 内 {} 次）",
                                 joint_name(j), stats[j].worst_q * rad2deg,
                                 limits_[j].first * rad2deg, limits_[j].second * rad2deg, stats[j].count);
                }
                if (dropped > 0) {
                    spdlog::warn("JointRecorder 缓冲已满，近 1s 内丢弃 {} 条采样。", dropped);
                }
            }
        }
        close_file();
    }

    std::vector<std::string> names_;
    std::vector<std::pair<float, float>> limits_;

    std::mutex mutex_;
    std::vector<Sample> pending_;
    std::array<LimitStat, kMaxJoints> limit_stats_{};
    size_t dropped_ = 0;
    std::vector<std::string> session_names_;
    size_t session_ = 0;
    clock::time_point session_start_;
    bool active_ = false;

    std::atomic<bool> stop_{false};
    std::thread worker_;
};
