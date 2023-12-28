#include "miniz.hpp"
#include <miniz.h>

#include <stdexcept>

class ZipArchiveImpl {
public:
    mz_zip_archive _archive = {0};
};


ZipArchive::ZipArchive(const std::vector<char>& buffer): pImpl(new ZipArchiveImpl()) {
    mz_bool status;
    status = mz_zip_reader_init_mem(
        &this->pImpl->_archive, buffer.data(), buffer.size(), 0
    );
    if(!status)
        throw new std::runtime_error("Could not initialize archive");
}

ZipArchive::~ZipArchive() {
    mz_zip_reader_end(&this->pImpl->_archive);
    delete this->pImpl;
}

size_t ZipArchive::get_number_of_files() const {
    return (size_t) mz_zip_reader_get_num_files(&this->pImpl->_archive);
}

std::vector<std::string> ZipArchive::get_file_paths() const {
    const size_t n = this->get_number_of_files();
    std::vector<std::string> result;
    for(int i = 0; i < n; i++){
        mz_zip_archive_file_stat file_stat;
        if (!mz_zip_reader_file_stat(&this->pImpl->_archive, i, &file_stat))
            throw new std::runtime_error("Cannot get info about file "+i);
        
        result.push_back(file_stat.m_filename);
    }
    return result;
}

std::vector<uint8_t> ZipArchive::read_file(const std::string& path) const {
    size_t uncomp_size;
    void* p = mz_zip_reader_extract_file_to_heap(
        &this->pImpl->_archive, path.c_str(), &uncomp_size, 0
    );
    if (!p) {
        throw new std::runtime_error("Could not read file "+path);
    }

    const std::vector<uint8_t> result(
        reinterpret_cast<const uint8_t*>(p),
        reinterpret_cast<const uint8_t*>(p) + uncomp_size
    );
    mz_free(p);
    return result;
}
