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
 * 两条互不依赖的后台线程：
 * - 终端线程：程序启动即创建。有超限或丢样时，每秒最多打印一条汇总。
 * - 记录线程：仅在命令行带 -r/--record 时创建。从 start() 起把整次运行写入同一个 csv，
 *   用 mode 列区分 Passive / FixStand / Velocity 等阶段。
  *
 * sample() 在控制线程里调用，只做拷贝。关节顺序为 SDK 电机顺序，
 * 关节名与限位来自 config.yaml 的 joint_names / joint_pos_limits。
 * q_des 取本拍即将下发的电机位置指令（AB 转换之前）。
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

    static bool started()
    {
        return started_flag().load(std::memory_order_acquire);
    }

    /** 程序进入控制循环前调用。终端线程总会启动；记录线程仅在 param::record 时启动。 */
    void start()
    {
        bool expected = false;
        if (!started_flag().compare_exchange_strong(expected, true)) return;

        spdlog::default_logger();
        t0_ = clock::now();
        recording_.store(param::record, std::memory_order_release);
        if (recording_.load(std::memory_order_relaxed)) {
            std::filesystem::create_directories(param::record_dir);
            file_path_ = param::record_dir / (timestamp() + ".csv");
        }

        console_thread_ = std::thread(&JointRecorder::console_loop, this);
        if (recording_.load(std::memory_order_relaxed)) {
            record_thread_ = std::thread(&JointRecorder::record_loop, this);
            spdlog::info("关节数据记录已开启，输出: {}", file_path_.string());
        }
    }

    /** 停止采样、刷完 csv、回收两条线程。应在控制线程退出后调用。 */
    void shutdown()
    {
        stop_.store(true, std::memory_order_release);
        started_flag().store(false, std::memory_order_release);
        if (console_thread_.joinable()) console_thread_.join();
        if (record_thread_.joinable()) record_thread_.join();
    }

    /**
     * 每个控制周期调用一次。未 start() 时直接返回。
     * @param mode 当前 FSM 状态名
     */
    template <typename LowStatePtr, typename LowCmdPtr>
    void sample(const LowStatePtr& lowstate, const LowCmdPtr& lowcmd, const std::string& mode)
    {
        if (!started_flag().load(std::memory_order_acquire)) return;

        Sample s;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        s.q_des.fill(nan);
        s.q.fill(nan);
        s.dq.fill(nan);
        s.tau.fill(nan);
        s.mode = mode;
        const int n_cfg = names_.empty() ? kMaxJoints : static_cast<int>(names_.size());
        {
            const auto& cmds = lowcmd->msg_.motor_cmd();
            s.n = std::min(n_cfg, static_cast<int>(cmds.size()));
            for (int j = 0; j < s.n; ++j) s.q_des[j] = cmds[j].q();
        }
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            const auto& motors = lowstate->msg_.motor_state();
            s.n = std::min(s.n, static_cast<int>(motors.size()));
            for (int j = 0; j < s.n; ++j) {
                s.q[j] = motors[j].q();
                s.dq[j] = motors[j].dq();
                s.tau[j] = motors[j].tau_est();
            }
            s.tick = lowstate->msg_.tick();
        }
        s.t = std::chrono::duration<double>(clock::now() - t0_).count();

        std::lock_guard<std::mutex> lock(mutex_);
        for (int j = 0; j < s.n && j < static_cast<int>(limits_.size()); ++j) {
            const float q = s.q_des[j];
            const auto& [lo, hi] = limits_[j];
            if (std::isnan(q) || (q >= lo && q <= hi)) continue;
            const float exceed = q < lo ? lo - q : q - hi;
            auto& st = limit_stats_[j];
            if (st.count == 0 || exceed > st.worst_exceed) {
                st.worst_exceed = exceed;
                st.worst_q = q;
                st.mode = mode;
            }
            ++st.count;
        }

        if (!recording_.load(std::memory_order_relaxed)) return;
        if (pending_.size() < kMaxPending) {
            pending_.push_back(std::move(s));
        } else {
            ++dropped_;
        }
    }

    ~JointRecorder()
    {
        shutdown();
    }

private:
    using clock = std::chrono::steady_clock;
    static constexpr size_t kMaxPending = 8192;

    struct Sample {
        double t = 0;
        uint32_t tick = 0;
        int n = 0;
        std::string mode;
        std::array<float, kMaxJoints> q_des{}, q{}, dq{}, tau{};
    };

    struct LimitStat {
        int count = 0;
        float worst_q = 0.0f;
        float worst_exceed = 0.0f;
        std::string mode;
    };

    static std::atomic<bool>& started_flag()
    {
        static std::atomic<bool> flag{false};
        return flag;
    }

    JointRecorder()
    {
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
            spdlog::warn("config.yaml 未配置 joint_pos_limits，已禁用期望位置限位检查。");
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

    void write_header(int n)
    {
        fmt::format_to(std::back_inserter(buf_), "t,tick,mode");
        for (int j = 0; j < n; ++j) {
            auto name = joint_name(j);
            fmt::format_to(std::back_inserter(buf_), ",{0}_q_des,{0}_q,{0}_dq,{0}_tau_est", name);
        }
        buf_.push_back('\n');
        header_n_ = n;
    }

    void write_sample(const Sample& s)
    {
        fmt::format_to(std::back_inserter(buf_), "{:.4f},{},{}", s.t, s.tick, s.mode);
        const int n = header_n_ > 0 ? header_n_ : s.n;
        for (int j = 0; j < n; ++j) {
            fmt::format_to(std::back_inserter(buf_), ",{:.6g},{:.6g},{:.6g},{:.6g}",
                           s.q_des[j], s.q[j], s.dq[j], s.tau[j]);
        }
        buf_.push_back('\n');
    }

    void flush_buf()
    {
        if (!ofs_.is_open() || buf_.size() == 0) return;
        ofs_.write(buf_.data(), static_cast<std::streamsize>(buf_.size()));
        ofs_.flush();
        buf_.clear();
    }

    void record_loop()
    {
        ofs_.open(file_path_);
        if (!ofs_) {
            spdlog::error("无法打开关节记录文件 {}", file_path_.string());
            return;
        }
        if (!names_.empty()) write_header(static_cast<int>(names_.size()));

        std::vector<Sample> batch;
        batch.reserve(kMaxPending);
        while (!stop_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            drain(batch);
        }
        drain(batch);
        flush_buf();
        if (ofs_.is_open()) {
            ofs_.close();
            spdlog::info("关节数据已保存至 {}", file_path_.string());
        }
    }

    void drain(std::vector<Sample>& batch)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batch.swap(pending_);
        }
        for (const auto& s : batch) {
            if (header_n_ == 0) write_header(s.n);
            write_sample(s);
        }
        batch.clear();
        flush_buf();
    }

    /** 有内容才打印，且两次打印至少间隔 1s。退出时再补打最后不足 1s 的汇总。 */
    void console_loop()
    {
        auto next = clock::now() + std::chrono::seconds(1);
        while (!stop_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (clock::now() < next) continue;
            report_once();
            next = clock::now() + std::chrono::seconds(1);
        }
        report_once();
    }

    void report_once()
    {
        std::array<LimitStat, kMaxJoints> stats{};
        size_t dropped = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats = limit_stats_;
            limit_stats_ = {};
            dropped = dropped_;
            dropped_ = 0;
        }

        std::string detail;
        int nbad = 0;
        std::string mode;
        bool mixed = false;
        constexpr float rad2deg = 180.0f / static_cast<float>(M_PI);
        for (int j = 0; j < kMaxJoints && j < static_cast<int>(limits_.size()); ++j) {
            if (stats[j].count == 0) continue;
            if (nbad == 0) mode = stats[j].mode;
            else if (stats[j].mode != mode) mixed = true;
            if (nbad) detail += "；";
            detail += fmt::format("{} {:.1f} deg 超出 [{:.1f}, {:.1f}]（{} 次）",
                                  joint_name(j), stats[j].worst_q * rad2deg,
                                  limits_[j].first * rad2deg, limits_[j].second * rad2deg,
                                  stats[j].count);
            ++nbad;
        }
        if (nbad == 0 && dropped == 0) return;

        if (nbad > 0 && dropped > 0) {
            spdlog::warn("[{}] 期望位置超限 {} 个关节（近 1s）：{}；记录缓冲丢弃 {} 条",
                         mixed ? "多模式" : mode, nbad, detail, dropped);
        } else if (nbad > 0) {
            spdlog::warn("[{}] 期望位置超限 {} 个关节（近 1s）：{}",
                         mixed ? "多模式" : mode, nbad, detail);
        } else {
            spdlog::warn("关节记录缓冲已满，近 1s 丢弃 {} 条采样。", dropped);
        }
    }

    std::vector<std::string> names_;
    std::vector<std::pair<float, float>> limits_;

    std::mutex mutex_;
    std::vector<Sample> pending_;
    std::array<LimitStat, kMaxJoints> limit_stats_{};
    size_t dropped_ = 0;
    clock::time_point t0_{};
    std::atomic<bool> recording_{false};
    std::atomic<bool> stop_{false};

    std::thread console_thread_;
    std::thread record_thread_;
    std::filesystem::path file_path_;
    std::ofstream ofs_;
    fmt::memory_buffer buf_;
    int header_n_ = 0;
};
