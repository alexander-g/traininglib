#pragma once

#include <vector>
#include <string>


class ZipArchiveImpl;

class ZipArchive {
    public:
    ZipArchive(const std::vector<char>& buffer);
    ~ZipArchive();

    size_t                   get_number_of_files() const;
    std::vector<std::string> get_file_paths() const;
    std::vector<uint8_t>     read_file(const std::string& path) const;

    private:
    ZipArchiveImpl* pImpl;

};
