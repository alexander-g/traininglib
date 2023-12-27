#pragma once

#include <vector>
#include <string>


class ZipArchiveImpl;

class ZipArchive {
    public:
    ZipArchive(const std::vector<char>& buffer);
    ~ZipArchive();

    size_t get_number_of_files();
    std::vector<std::string> get_file_paths();

    private:
    ZipArchiveImpl* pImpl;

};
