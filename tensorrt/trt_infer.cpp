#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CHECK(status)                                                          \
    do {                                                                       \
        const auto ret = (status);                                             \
        if (ret != 0) {                                                        \
            std::cerr << "CUDA failure: " << ret << std::endl;                 \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

static std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening engine file: " << path << std::endl;
        std::abort();
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading engine file" << std::endl;
        std::abort();
    }
    return buffer;
}

static int64_t volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return -1;
        }
        v *= static_cast<int64_t>(dims.d[i]);
    }
    return v;
}

static nvinfer1::Dims makeDims(std::initializer_list<int> values) {
    nvinfer1::Dims d{};
    d.nbDims = static_cast<int>(values.size());
    int i = 0;
    for (int v : values) {
        d.d[i++] = v;
    }
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine_path> <batch_size> [obs_horizon=2] [num_cameras=2] "
                  << "[image_size=84] [proprio_dim=7] [iterations=100]" << std::endl;
        return -1;
    }

    const std::string enginePath = argv[1];
    const int batchSize = std::stoi(argv[2]);
    const int obsHorizon = (argc > 3) ? std::stoi(argv[3]) : 2;
    const int numCameras = (argc > 4) ? std::stoi(argv[4]) : 2;
    const int imageSize = (argc > 5) ? std::stoi(argv[5]) : 84;
    const int proprioDim = (argc > 6) ? std::stoi(argv[6]) : 7;
    const int numIters = (argc > 7) ? std::stoi(argv[7]) : 100;
    const int warmupIters = 10;

    std::cout << "Loading TensorRT engine: " << enginePath << std::endl;
    const auto engineData = loadEngineFile(enginePath);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return -1;
    }
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (engine == nullptr) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        delete runtime;
        return -1;
    }
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (context == nullptr) {
        std::cerr << "Failed to create execution context" << std::endl;
        delete engine;
        delete runtime;
        return -1;
    }

    const char* imagesName = "images";
    const char* proprioName = "proprio";
    const char* outputName = "pred_actions";

    const nvinfer1::Dims imagesDims = makeDims({batchSize, obsHorizon, numCameras, 3, imageSize, imageSize});
    const nvinfer1::Dims proprioDims = makeDims({batchSize, obsHorizon, proprioDim});

    if (!context->setInputShape(imagesName, imagesDims)) {
        std::cerr << "Failed to set input shape for " << imagesName << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return -1;
    }
    if (!context->setInputShape(proprioName, proprioDims)) {
        std::cerr << "Failed to set input shape for " << proprioName << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return -1;
    }

    const nvinfer1::Dims outputDims = context->getTensorShape(outputName);
    const int64_t outElems = volume(outputDims);
    if (outElems <= 0) {
        std::cerr << "Invalid output shape for " << outputName << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return -1;
    }

    const size_t imagesElems = static_cast<size_t>(batchSize) * obsHorizon * numCameras * 3 * imageSize * imageSize;
    const size_t proprioElems = static_cast<size_t>(batchSize) * obsHorizon * proprioDim;
    const size_t outputElems = static_cast<size_t>(outElems);

    std::vector<float> hImages(imagesElems);
    std::vector<float> hProprio(proprioElems);
    std::vector<float> hOutput(outputElems);

    std::mt19937 rng(42);
    std::normal_distribution<float> normalDist(0.0f, 1.0f);
    for (auto& x : hImages) {
        x = normalDist(rng);
    }
    for (auto& x : hProprio) {
        x = normalDist(rng);
    }

    void* dImages = nullptr;
    void* dProprio = nullptr;
    void* dOutput = nullptr;
    const size_t imagesBytes = imagesElems * sizeof(float);
    const size_t proprioBytes = proprioElems * sizeof(float);
    const size_t outputBytes = outputElems * sizeof(float);

    CHECK(cudaMalloc(&dImages, imagesBytes));
    CHECK(cudaMalloc(&dProprio, proprioBytes));
    CHECK(cudaMalloc(&dOutput, outputBytes));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    if (!context->setTensorAddress(imagesName, dImages) ||
        !context->setTensorAddress(proprioName, dProprio) ||
        !context->setTensorAddress(outputName, dOutput)) {
        std::cerr << "Failed to set one or more tensor addresses" << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(dImages);
        cudaFree(dProprio);
        cudaFree(dOutput);
        delete context;
        delete engine;
        delete runtime;
        return -1;
    }

    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < warmupIters; ++i) {
        CHECK(cudaMemcpyAsync(dImages, hImages.data(), imagesBytes, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(dProprio, hProprio.data(), proprioBytes, cudaMemcpyHostToDevice, stream));
        if (!context->enqueueV3(stream)) {
            std::cerr << "enqueueV3 failed during warmup" << std::endl;
            cudaStreamDestroy(stream);
            cudaFree(dImages);
            cudaFree(dProprio);
            cudaFree(dOutput);
            delete context;
            delete engine;
            delete runtime;
            return -1;
        }
        CHECK(cudaMemcpyAsync(hOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
    }

    std::cout << "Running benchmark..." << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIters; ++i) {
        CHECK(cudaMemcpyAsync(dImages, hImages.data(), imagesBytes, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(dProprio, hProprio.data(), proprioBytes, cudaMemcpyHostToDevice, stream));
        if (!context->enqueueV3(stream)) {
            std::cerr << "enqueueV3 failed during benchmark" << std::endl;
            cudaStreamDestroy(stream);
            cudaFree(dImages);
            cudaFree(dProprio);
            cudaFree(dOutput);
            delete context;
            delete engine;
            delete runtime;
            return -1;
        }
        CHECK(cudaMemcpyAsync(hOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream));
    }
    CHECK(cudaStreamSynchronize(stream));
    const auto end = std::chrono::high_resolution_clock::now();

    const double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    const double avgMs = totalMs / static_cast<double>(numIters);
    const double throughput = (batchSize * 1000.0) / avgMs;

    std::cout << "=====================================" << std::endl;
    std::cout << "Task: Robomimic policy inference" << std::endl;
    std::cout << "Batch Size: " << batchSize << std::endl;
    std::cout << "Input images shape: (" << batchSize << "," << obsHorizon << "," << numCameras
              << ",3," << imageSize << "," << imageSize << ")" << std::endl;
    std::cout << "Input proprio shape: (" << batchSize << "," << obsHorizon << "," << proprioDim << ")" << std::endl;
    std::cout << "Output pred_actions dims:";
    for (int i = 0; i < outputDims.nbDims; ++i) {
        std::cout << (i == 0 ? " (" : ",") << outputDims.d[i];
    }
    std::cout << ")" << std::endl;
    std::cout << "Avg Latency: " << avgMs << " ms" << std::endl;
    std::cout << "Throughput: " << throughput << " sequences/sec" << std::endl;
    std::cout << "=====================================" << std::endl;

    // Preview first predicted action for batch element 0, horizon step 0.
    if (outputDims.nbDims == 3 && outputDims.d[1] > 0 && outputDims.d[2] > 0) {
        const int actionDim = outputDims.d[2];
        std::cout << "Pred action[0,0,:] = [";
        for (int i = 0; i < actionDim; ++i) {
            const float v = hOutput[static_cast<size_t>(i)];
            std::cout << v;
            if (i + 1 < actionDim) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(dImages));
    CHECK(cudaFree(dProprio));
    CHECK(cudaFree(dOutput));
    delete context;
    delete engine;
    delete runtime;
    return 0;
}
