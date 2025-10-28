#include "miniz.hpp"
#include <miniz.h>

#include <stdexcept>

class ZipArchiveImpl {
public:
    mz_zip_archive _archive = {0};
};


ZipArchive::ZipArchive(const char* data, size_t datasize): 
    pImpl(new ZipArchiveImpl()){
    
    mz_bool status;
    status = mz_zip_reader_init_mem(&this->pImpl->_archive, data, datasize, 0);
    if(!status)
        throw new std::runtime_error("Could not initialize archive");
}

ZipArchive::ZipArchive(const std::vector<char>& buffer): 
    ZipArchive(buffer.data(), buffer.size()) {}

ZipArchive::ZipArchive(): pImpl(new ZipArchiveImpl()) {
    mz_bool status = mz_zip_writer_init_heap(&this->pImpl->_archive, 0, 1024);
    if(!status)
        throw new std::runtime_error("Could not initialize archive");
}

ZipArchive::~ZipArchive() {
    mz_zip_reader_end(&this->pImpl->_archive);
    mz_zip_writer_end(&this->pImpl->_archive);
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

void ZipArchive::write_file(
    const std::string& path, 
    const void*        pbuffer, 
    size_t             buffersize
) {
    const mz_bool status = mz_zip_writer_add_mem(
        &this->pImpl->_archive,
        path.c_str(),
        pbuffer,
        buffersize,
        MZ_NO_COMPRESSION
    );
    if(!status) {
        throw new std::runtime_error("Could not write to archive.");
    }
}


std::vector<uint8_t> ZipArchive::to_bytes() {
    void*  p = 0;
    size_t buffersize;
    const mz_bool status = mz_zip_writer_finalize_heap_archive(
        &this->pImpl->_archive, &p, &buffersize
    );
    if(!status || !p) {
        throw new std::runtime_error("Could not finalize archive.");
    }

    const std::vector<uint8_t> result(
        reinterpret_cast<const uint8_t*>(p),
        reinterpret_cast<const uint8_t*>(p) + buffersize
    );
    mz_free(p);
    return result;
}
