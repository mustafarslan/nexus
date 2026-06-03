#include "nexus_page_mounter.hpp"
#include "nexus_os_compat.hpp"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <thread>

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#define open _open
#define fstat _fstat
#define close _close
#define stat _stat
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

NexusPageMounter::NexusPageMounter(const std::string& atb_filepath) {
    fd_ = open(atb_filepath.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("NexusPageMounter: Failed to open ATB file: " + atb_filepath);
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("NexusPageMounter: Failed to stat ATB file: " + atb_filepath);
    }
    file_size_ = sb.st_size;

    if (file_size_ < sizeof(AeonToolBlockHeader)) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("NexusPageMounter: ATB file is smaller than the header size: " + atb_filepath);
    }

    mapped_data_ = map_file(fd_, file_size_);
    if (!mapped_data_) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("NexusPageMounter: Failed to mmap ATB file: " + atb_filepath);
    }

    // Prefetch the pages to advise the OS to load them into RAM
    prefetch_memory(mapped_data_, file_size_);

    header_ = static_cast<const AeonToolBlockHeader*>(mapped_data_);

    // Validate ATB header properties
    if (std::memcmp(header_->magic, "ATB1", 4) != 0) {
        unmap_file(mapped_data_, file_size_);
        close(fd_);
        fd_ = -1;
        mapped_data_ = nullptr;
        header_ = nullptr;
        throw std::runtime_error("NexusPageMounter: Invalid ATB magic bytes in file: " + atb_filepath);
    }

    if (header_->version != 1) {
        unmap_file(mapped_data_, file_size_);
        close(fd_);
        fd_ = -1;
        mapped_data_ = nullptr;
        header_ = nullptr;
        throw std::runtime_error("NexusPageMounter: Unsupported ATB version in file: " + atb_filepath);
    }

    if (header_->k_tensor_offset + header_->k_total_bytes > file_size_ ||
        header_->v_tensor_offset + header_->v_total_bytes > file_size_) {
        unmap_file(mapped_data_, file_size_);
        close(fd_);
        fd_ = -1;
        mapped_data_ = nullptr;
        header_ = nullptr;
        throw std::runtime_error("NexusPageMounter: Corrupted ATB offsets exceed file size: " + atb_filepath);
    }
}

NexusPageMounter::~NexusPageMounter() {
    if (mapped_data_) {
        unmap_file(mapped_data_, file_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

std::span<const uint8_t> NexusPageMounter::get_k_tensors() const {
    if (!mapped_data_ || !header_) return {};
    const uint8_t* ptr = static_cast<const uint8_t*>(mapped_data_) + header_->k_tensor_offset;
    return { ptr, static_cast<size_t>(header_->k_total_bytes) };
}

std::span<const uint8_t> NexusPageMounter::get_v_tensors() const {
    if (!mapped_data_ || !header_) return {};
    const uint8_t* ptr = static_cast<const uint8_t*>(mapped_data_) + header_->v_tensor_offset;
    return { ptr, static_cast<size_t>(header_->v_total_bytes) };
}

