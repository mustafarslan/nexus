#pragma once
#include <cstdint>
#include <string>
#include <span>
#include "aeon_tool_block.hpp"

class NexusPageMounter {
private:
    int fd_ = -1;
    size_t file_size_ = 0;
    void* mapped_data_ = nullptr;
    const AeonToolBlockHeader* header_ = nullptr;

public:
    explicit NexusPageMounter(const std::string& atb_filepath);
    ~NexusPageMounter();

    // Disable copy
    NexusPageMounter(const NexusPageMounter&) = delete;
    NexusPageMounter& operator=(const NexusPageMounter&) = delete;

    const AeonToolBlockHeader* get_header() const { return header_; }
    void* get_mapped_data() const { return mapped_data_; }
    
    // Returns read-only spans directly to the page cache
    std::span<const uint8_t> get_k_tensors() const;
    std::span<const uint8_t> get_v_tensors() const;
};
