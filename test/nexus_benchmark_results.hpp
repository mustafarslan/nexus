#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/utsname.h>
#include <vector>

namespace nexus::bench {

struct Series {
    std::string name;
    std::string unit;
    std::vector<double> samples;
};

inline std::string shell_capture(const char * command) {
    std::array<char, 256> buffer{};
    std::string result;
    FILE * pipe = popen(command, "r");
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result;
}

inline std::string json_escape(const std::string & input) {
    std::ostringstream out;
    for (char c : input) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out << "\\u" << std::hex << static_cast<int>(c);
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

inline std::string quote(const std::string & value) {
    return "\"" + json_escape(value) + "\"";
}

inline std::string iso8601_utc_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buffer;
}

inline std::string hardware_label() {
    struct utsname info {};
    if (uname(&info) != 0) {
        return "unknown";
    }
    return std::string(info.sysname) + " " + info.release + " " + info.machine;
}

inline double percentile(std::vector<double> sorted, double p) {
    if (sorted.empty()) {
        return 0.0;
    }
    std::sort(sorted.begin(), sorted.end());
    double rank = p * static_cast<double>(sorted.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(rank));
    size_t hi = static_cast<size_t>(std::ceil(rank));
    if (lo == hi) {
        return sorted[lo];
    }
    double frac = rank - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

inline void write_summary(std::ostream & out, const std::vector<double> & samples) {
    double mean = 0.0;
    double stddev = 0.0;
    if (!samples.empty()) {
        mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        double sq_sum = 0.0;
        for (double v : samples) {
            double d = v - mean;
            sq_sum += d * d;
        }
        stddev = samples.size() > 1 ? std::sqrt(sq_sum / static_cast<double>(samples.size() - 1)) : 0.0;
    }

    out << "{"
        << "\"mean\":" << mean
        << ",\"std\":" << stddev
        << ",\"p50\":" << percentile(samples, 0.50)
        << ",\"p90\":" << percentile(samples, 0.90)
        << ",\"p99\":" << percentile(samples, 0.99)
        << ",\"n\":" << samples.size()
        << "}";
}

inline void write_artifact(
    const std::filesystem::path & output_path,
    const std::string & benchmark_name,
    const std::string & model_hash,
    const std::map<std::string, std::string> & config,
    const std::vector<Series> & series) {

    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("failed to open benchmark artifact: " + output_path.string());
    }

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"benchmark\": " << quote(benchmark_name) << ",\n";
    out << "  \"git_sha\": " << quote(shell_capture("git rev-parse HEAD 2>/dev/null")) << ",\n";
    out << "  \"llama_cpp_sha\": " << quote(shell_capture("git -C external/llama.cpp rev-parse HEAD 2>/dev/null")) << ",\n";
    out << "  \"model_hash\": " << quote(model_hash) << ",\n";
    out << "  \"hardware\": " << quote(hardware_label()) << ",\n";
    out << "  \"timestamp\": " << quote(iso8601_utc_now()) << ",\n";
    out << "  \"config\": {";
    bool first = true;
    for (const auto & [key, value] : config) {
        if (!first) out << ",";
        out << "\n    " << quote(key) << ": " << quote(value);
        first = false;
    }
    if (!config.empty()) out << "\n  ";
    out << "},\n";
    out << "  \"metrics\": {\n";
    for (size_t i = 0; i < series.size(); ++i) {
        const auto & s = series[i];
        out << "    " << quote(s.name) << ": {\n";
        out << "      \"unit\": " << quote(s.unit) << ",\n";
        out << "      \"raw_samples\": [";
        for (size_t j = 0; j < s.samples.size(); ++j) {
            if (j > 0) out << ", ";
            out << s.samples[j];
        }
        out << "],\n";
        out << "      \"summary\": ";
        write_summary(out, s.samples);
        out << "\n    }";
        if (i + 1 < series.size()) out << ",";
        out << "\n";
    }
    out << "  }\n";
    out << "}\n";
}

} // namespace nexus::bench
