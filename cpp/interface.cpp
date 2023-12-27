#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "miniz.hpp"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <torch/script.h>



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

    int initialize_module(const uint8_t* pbuffer, const size_t buffersize) {
        try {
            std::istringstream stream(
                std::string(reinterpret_cast<const char*>(pbuffer), buffersize)
            );
            torch::jit::script::Module module = torch::jit::load(stream);
            return 0;
        } catch (const std::exception& e) {
            return 1;
        }
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


