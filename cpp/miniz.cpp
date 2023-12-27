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

size_t ZipArchive::get_number_of_files() {
    return (size_t) mz_zip_reader_get_num_files(&this->pImpl->_archive);
}

std::vector<std::string> ZipArchive::get_file_paths() {
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

