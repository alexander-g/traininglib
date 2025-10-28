#pragma once

#include <vector>
#include <string>


class ZipArchiveImpl;

class ZipArchive {
    public:
    /** New archive for writing */
    ZipArchive();
    /** Archive for reading from existing */
    ZipArchive(const std::vector<char>& buffer);
    ZipArchive(const char* data, size_t datasize);
    ~ZipArchive();

    size_t                   get_number_of_files() const;
    std::vector<std::string> get_file_paths() const;
    std::vector<uint8_t>     read_file(const std::string& path) const;
    void                     write_file(
        const std::string& path, const void* pbuffer, size_t buffersize
    );
    std::vector<uint8_t>     to_bytes();

    private:
    ZipArchiveImpl* pImpl;

};
