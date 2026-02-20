#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <cmath>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort(); \
        } \
    } while (0)

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening engine file: " << path << std::endl;
        abort();
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading engine file" << std::endl;
        abort();
    }
    return buffer;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <batch_size>" << std::endl;
        return -1;
    }
    
    std::string enginePath = argv[1];
    int BATCH_SIZE = std::stoi(argv[2]);
    const int N_CELLS = 81;
    const int D_VOCAB = 10;
    
    std::cout << "Loading TensorRT Engine..." << std::endl;
    auto engineData = loadEngineFile(enginePath);
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    
    // Set dynamic shapes
    nvinfer1::Dims z_dims;
    z_dims.nbDims = 3; z_dims.d[0] = BATCH_SIZE; z_dims.d[1] = 1; z_dims.d[2] = N_CELLS;
    context->setInputShape("z", z_dims);
    
    nvinfer1::Dims m_dims;
    m_dims.nbDims = 2; m_dims.d[0] = BATCH_SIZE; m_dims.d[1] = N_CELLS;
    context->setInputShape("M", m_dims);
    
    // Allocate host memory
    size_t z_size = BATCH_SIZE * 1 * N_CELLS * sizeof(int64_t);
    size_t m_size = BATCH_SIZE * N_CELLS * sizeof(bool); // bool in ONNX maps to 1 byte C++
    size_t out_size = BATCH_SIZE * 1 * N_CELLS * D_VOCAB * sizeof(float);
    
    std::vector<int64_t> h_z(BATCH_SIZE * N_CELLS);
    std::vector<uint8_t> h_m(BATCH_SIZE * N_CELLS); // using uint8_t for bool to get raw data pointer
    std::vector<float> h_out(BATCH_SIZE * N_CELLS * D_VOCAB);
    
    // Load board and mask data from binary export script
    std::ifstream data_file("../sudoku_test_boards.bin", std::ios::binary);
    if (!data_file) {
        std::cerr << "Error opening data file! Did you run export_test_data.py?" << std::endl;
        abort();
    }
    
    data_file.read(reinterpret_cast<char*>(h_z.data()), BATCH_SIZE * N_CELLS * sizeof(int64_t));
    data_file.read(reinterpret_cast<char*>(h_m.data()), BATCH_SIZE * N_CELLS * sizeof(uint8_t));
    
    std::cout << "Successfully loaded " << BATCH_SIZE << " real generated Sudoku boards!" << std::endl;
    
    // Allocate device memory
    void* d_z;
    void* d_m;
    void* d_out;
    CHECK(cudaMalloc(&d_z, z_size));
    CHECK(cudaMalloc(&d_m, m_size));
    CHECK(cudaMalloc(&d_out, out_size));
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        CHECK(cudaMemcpyAsync(d_z, h_z.data(), z_size, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(d_m, h_m.data(), m_size, cudaMemcpyHostToDevice, stream));
        context->setTensorAddress("z", d_z);
        context->setTensorAddress("M", d_m);
        context->setTensorAddress("z_hat", d_out);
        context->enqueueV3(stream);
        CHECK(cudaMemcpyAsync(h_out.data(), d_out, out_size, cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }
    
    std::cout << "Running benchmark..." << std::endl;
    int num_iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iters; ++i) {
        CHECK(cudaMemcpyAsync(d_z, h_z.data(), z_size, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(d_m, h_m.data(), m_size, cudaMemcpyHostToDevice, stream));
        context->enqueueV3(stream);
        CHECK(cudaMemcpyAsync(h_out.data(), d_out, out_size, cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / num_iters;
    
    std::cout << "=====================================" << std::endl;
    std::cout << "Batch Size: " << BATCH_SIZE << std::endl;
    std::cout << "Avg Latency: " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (BATCH_SIZE * 1000.0) / avg_ms << " sequences/sec" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Verify accuracy metric behavior
    int correct_masks = 0;
    int total_masks = 0;
    
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int i = 0; i < N_CELLS; ++i) {
            int idx = b * N_CELLS + i;
            if (h_m[idx]) {
                total_masks++;
                
                int max_digit = 0;
                float max_val = -1e9;
                for (int v = 0; v < D_VOCAB; ++v) {
                    float val = h_out[idx * D_VOCAB + v];
                    if (val > max_val) {
                        max_val = val;
                        max_digit = v;
                    }
                }
                
                if (max_digit == h_z[idx]) {
                    correct_masks++;
                }
            }
        }
    }
    
    std::cout << "Sanity check precision (randomized fake inputs): " 
              << (float)correct_masks / std::max(1, total_masks) * 100.0f << " %" << std::endl;
              
    cudaStreamDestroy(stream);
    cudaFree(d_z);
    cudaFree(d_m);
    cudaFree(d_out);
    delete context;
    delete engine;
    delete runtime;
    
    return 0;
}
