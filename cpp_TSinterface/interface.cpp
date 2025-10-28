#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "miniz.hpp"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <torch/script.h>


typedef torch::Dict<std::string, torch::Tensor> TensorDict;


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

bool endsWith(const std::string& str, const std::string& suffix) {
    return (
        str.size() >= suffix.size() 
        && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0
    );
}

json read_inference_schema_from_archive(const ZipArchive& archive){
    const std::vector<std::string> paths = archive.get_file_paths();
    for (const auto& path : paths) {
        if (endsWith(path, "/inference.schema.json")){
            const std::vector<uint8_t> data = archive.read_file(path);
            const json parsed = json::parse(data);
            return parsed;
        }
    }
    //else
    throw std::runtime_error("Could not find inference schema in archive.");
}

at::ScalarType string_to_dtype(const std::string& dtypestring) {
    if (dtypestring == "float32")
        return torch::kFloat32;
    else if (dtypestring == "float64")
        return torch::kFloat64;
    else if (dtypestring == "int64")
        return torch::kInt64;
    else if (dtypestring == "uint8")
        return torch::kUInt8;
    else if (dtypestring == "bool")
        return torch::kBool;
    else
        throw std::runtime_error("Unsupported dtype: " + dtypestring);
}

std::string dtype_to_string(const caffe2::TypeMeta& dtype) {
    if (dtype == torch::kFloat32)
        return "float32";
    else if (dtype == torch::kFloat64)
        return "float64";
    else if (dtype == torch::kInt64)
        return "int64";
    else if (dtype == torch::kUInt8)
        return "uint8";
    else if (dtype == torch::kBool)
        return "bool";
    else
        throw std::runtime_error("Unsupported dtype");
}

int64_t shape_to_size(const std::vector<int64_t> shape) {
    int64_t size = 1;
    for(const int64_t i: shape) {
        size *= i;
    }
    return size;
}

torch::Tensor
read_tensor_from_archive(const ZipArchive& archive, const json& schemaitem) {
    const auto path  = schemaitem.find("path");
    const auto dtype = schemaitem.find("dtype");
    const auto shape = schemaitem.find("shape");
    const auto end   = schemaitem.end();
    if(path == end || dtype == end || shape == end)
        throw std::runtime_error("Invalid schema item");
    
    std::vector<uint8_t> data = archive.read_file(*path);
    const std::vector<int64_t> shapevec = shape->get<std::vector<int64_t>>();
    //TODO: need to multiply with element size
    // if( shape_to_size(shapevec) != data.size() )
    //     throw std::runtime_error("Schema shape does not correspond to data");
    
    return torch::from_blob(data.data(), shapevec, string_to_dtype(*dtype)).clone();
}

TensorDict read_inputfeed_from_archive(const ZipArchive& archive) {
    const json schema = read_inference_schema_from_archive(archive);
    TensorDict inputfeed;
    for (auto item = schema.begin(); item != schema.end(); ++item) {
        const torch::Tensor t = read_tensor_from_archive(archive, item.value());
        inputfeed.insert(item.key(), t);
    }
    return inputfeed;
}

torch::Tensor create_tensorview_from_schema_item(const json& schemaitem) {
    const auto addr  = schemaitem.find("address");
    const auto dtype = schemaitem.find("dtype");
    const auto shape = schemaitem.find("shape");
    const auto end   = schemaitem.end();
    if(addr == end || dtype == end || shape == end)
        throw std::runtime_error("Invalid schema item");
    
    const std::vector<int64_t> shapevec = shape->get<std::vector<int64_t>>();
    void* address = (void*) addr->get<uint64_t>();
    return torch::from_blob(address, shapevec, string_to_dtype(*dtype));
}

TensorDict read_tensordict_from_json(const std::vector<uint8_t>& jsonbuffer) {
    const json schema = json::parse(jsonbuffer);
    TensorDict inputfeed;
    for (auto item = schema.begin(); item != schema.end(); ++item) {
        const torch::Tensor t = create_tensorview_from_schema_item(item.value());
        inputfeed.insert(item.key(), t);
    }
    return inputfeed;
}

std::vector<uint8_t> write_tensordict_to_archive(const TensorDict& data) {
    json schema({});
    ZipArchive archive;
    int i = 0;
    for(const auto& item: data){
        const torch::Tensor& t = item.value();
        const auto& key        = item.key();

        std::ostringstream oss;
        oss << "./data/" << i << ".storage";
        const std::string path = oss.str();

        archive.write_file(path, t.data_ptr(), t.nbytes());

        json schema_item;
        schema_item["path"]  = path;
        schema_item["dtype"] = dtype_to_string(t.dtype());
        schema_item["shape"] = t.sizes();
        schema[item.key()]   = schema_item;

        i++;
    }
    const std::string schema_str = schema.dump();
    archive.write_file(
        "./onnx/inference.schema.json", schema_str.c_str(), schema_str.size()
    );
    return archive.to_bytes();
}

TensorDict to_tensordict(const torch::jit::IValue& x) {
    if(!x.isGenericDict())
        throw std::runtime_error("Value is not a dict");
    
    const torch::Dict<c10::IValue, c10::IValue> x_dict = x.toGenericDict();
    torch::Dict<std::string, torch::Tensor> result;
    for (const auto& pair : x_dict) {
        if(!pair.key().isString() || !pair.value().isTensor())
            throw std::runtime_error("Dict is not a string-tensor dict.");

        result.insert(pair.key().toString()->string(), pair.value().toTensor());
    }
    return result;
}

void write_data_to_output(
    const std::vector<uint8_t>& data,
    uint8_t**                   outputbuffer,
    size_t*                     outputsize
) {
    *outputsize   = data.size();
    *outputbuffer = new uint8_t[data.size()];
    std::memcpy(*outputbuffer, data.data(), data.size());
}


void write_tensordict_to_outputbuffer(
    const TensorDict& data,
    uint8_t**         outputbuffer,
    size_t*           outputbuffersize
) {
    size_t datasize = 0;
    json schema({});
    for(const auto& item: data){
        const torch::Tensor& t = item.value();
        const auto& key        = item.key();

        datasize += t.nbytes();

        json schema_item;
        schema_item["address"] = 0xffffffffffffffff;  //placeholder
        schema_item["dtype"]   = dtype_to_string(t.dtype());
        schema_item["shape"]   = t.sizes();
        schema[key]            = schema_item;
    }
    std::string schema_str = schema.dump();

    uint8_t* buffer     = new uint8_t[datasize + schema_str.size()];
    size_t bytes_copied = 0;
    for(const auto& item: data){
        const torch::Tensor& t = item.value();
        const auto& key        = item.key();

        //copy the tensor data to the newly allocated buffer
        uint8_t* address       = buffer + schema_str.size() + bytes_copied;
        std::memcpy(address, t.data_ptr(), t.nbytes());
        bytes_copied          += t.nbytes();

        //update the address in the schema
        json schema_item       = schema[key];
        schema_item["address"] = (uint64_t) address;
        schema[key]            = schema_item;
    }
    //update the schema string with the new addresses
    schema_str = schema.dump();
    std::memcpy(buffer, schema_str.data(), schema_str.size() );

    *outputbuffer     = buffer;
    //tell caller only about json schema, the rest just hangs around until delete
    *outputbuffersize = schema_str.size();
}


void handle_eptr(std::exception_ptr eptr){
    try{
        if (eptr)
            std::rethrow_exception(eptr);
    } catch(const std::exception& e) {
        std::cout << "Caught exception: '" << e.what() << "'\n";
    }
}


//TODO: don't store this here
namespace global {
    torch::jit::script::Module module;
}

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

extern "C" {
    /** Initialize a TorchScript module from a buffer. */
    EXPORT int32_t initialize_module(const uint8_t* pbuffer, const size_t buffersize) {
        try {
            std::istringstream stream(
                std::string(reinterpret_cast<const char*>(pbuffer), buffersize)
            );
            global::module = torch::jit::load(stream);
            return 0;
        } catch (...) {
            return 1;
        }
    }

    EXPORT int32_t run_module(
        const uint8_t*  inputbuffer,
        const size_t    inputbuffersize,
              uint8_t** outputbuffer,
              size_t*   outputbuffersize,
              bool      debug = false
    ) {
        try {
            const auto t0{std::chrono::steady_clock::now()};
            const TensorDict inputfeed = read_tensordict_from_json(
                std::vector<uint8_t>(inputbuffer, inputbuffer+inputbuffersize)
            );
            const auto t1{std::chrono::steady_clock::now()};
            torch::jit::IValue output  = global::module.forward(
                {torch::IValue(inputfeed)}
            );
            const auto t2{std::chrono::steady_clock::now()};
            write_tensordict_to_outputbuffer(
                to_tensordict(output), outputbuffer, outputbuffersize
            );
            return 0;
        } catch (...) {
            if(debug)
                handle_eptr( std::current_exception() );
            return 1;
        }
    }

    EXPORT void free_memory(uint8_t* p) {
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


