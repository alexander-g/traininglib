#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <torch/script.h>


typedef torch::Dict<std::string, torch::Tensor> TensorDict;



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




