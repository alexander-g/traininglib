#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <miniz.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;


class ZipArchive {
    public:

    ZipArchive(const std::vector<char>& buffer) {
        mz_bool status;
        status = mz_zip_reader_init_mem(
            &this->_archive, buffer.data(), buffer.size(), 0
        );
        if(!status)
            throw new std::runtime_error("Could not initialize archive");
    }

    ~ZipArchive() {
        mz_zip_reader_end(&this->_archive);
    }

    size_t get_number_of_files() {
        return (size_t) mz_zip_reader_get_num_files(&this->_archive);
    }

    std::vector<std::string> get_file_paths() {
        const size_t n = this->get_number_of_files();
        std::vector<std::string> result;
        for(int i = 0; i < n; i++){
            mz_zip_archive_file_stat file_stat;
            if (!mz_zip_reader_file_stat(&this->_archive, i, &file_stat))
                throw new std::runtime_error("Cannot get info about file "+i);
            
            result.push_back(file_stat.m_filename);
        }
        return result;
    }

    private:
    mz_zip_archive _archive = {0};
};


std::vector<char> read_binary_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Error opening file: " + filename);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        return buffer;
    }
    else throw std::runtime_error("Error reading file: " + filename);
}



extern "C" {
    const char* banana(const uint8_t* pbuffer, const size_t buffersize) {
        const std::vector<char> buffer(pbuffer, pbuffer+buffersize);
        ZipArchive archive(buffer);
        const std::vector<std::string> paths = archive.get_file_paths();
        const json j_paths(paths);
        const std::string j_as_string = j_paths.dump();

        char* bytes = new char[j_as_string.length() + 1]; // +1 for null
        std::strcpy(bytes, j_as_string.c_str());
        return bytes;
    }

    void free_banana(const char* p) {
        delete[] p;
    }
}


int main() {
    const std::vector<char> buffer = read_binary_file("DELETE.zip");
    ZipArchive archive(buffer);

    std::cout << archive.get_number_of_files() << std::endl;
    const std::vector<std::string> file_paths = archive.get_file_paths();
    for (const auto& path : file_paths) {
        std::cout << path << std::endl;
    }

    std::cout<<"ok"<<std::endl;
}


