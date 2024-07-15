module app;

import std.stdio;
import std.getopt;
import core.stdc.stdlib;

import onnxruntime_c_api;

/* this demo code is based on:
- ORT C/FNSCANDY demo: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/fns_candy_style_transfer/fns_candy_style_transfer.c
- ORT C++/MNIST demo: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/MNIST/MNIST.cpp
*/

void ort_abort_on_error(const(OrtApi)* g_ort, OrtStatus* status) {
    if (status !is null) {
        const char* msg = g_ort.GetErrorMessage(status);
        stderr.writefln("%s", msg);
        g_ort.ReleaseStatus(status);
        abort();
    }
}

// template <typename T>
// static void softmax(T& input) {
//   float rowmax = *std::max_element(input.begin(), input.end());
//   std::vector<float> y(input.size());
//   float sum = 0.0f;
//   for (size_t i = 0; i != input.size(); ++i) {
//     sum += y[i] = std::exp(input[i] - rowmax);
//   }
//   for (size_t i = 0; i != input.size(); ++i) {
//     input[i] = y[i] / sum;
//   }
// }
void softmax(T)(ref T input) {
    float rowmax = float.min;
    for (size_t i = 0; i < input.length; i++) {
        if (input[i] > rowmax) {
            rowmax = input[i];
        }
    }
    float[] y = new float[input.length];
    float sum = 0.0f;
    for (size_t i = 0; i < input.length; i++) {
        sum += y[i] = exp(input[i] - rowmax);
    }
    for (size_t i = 0; i < input.length; i++) {
        input[i] = y[i] / sum;
    }
}

int main(string[] args) {
    writefln("loading onnxruntime v%s", ORT_API_VERSION);
    const(OrtApi)* g_ort = OrtGetApiBase().GetApi(ORT_API_VERSION);
    if (g_ort is null) {
        writefln("failed to load onnxruntime v%s", ORT_API_VERSION);
        return -1;
    }

    string mnist_model_path = "./mnist.onnx";
    string execution_provider = "cpu";
    auto help = getopt(args,
        "provider|p", &execution_provider,
    );
    if (help.helpWanted) {
        defaultGetoptPrinter("Usage: app [options]\n", help.options);
        return 1;
    }

    writefln("creating env");
    OrtEnv* env;
    OrtStatus* status = g_ort.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
    ort_abort_on_error(g_ort, status);
    assert(env !is null, "failed to create env");

    writefln("creating session options");
    OrtSessionOptions* session_options;
    status = g_ort.CreateSessionOptions(&session_options);
    ort_abort_on_error(g_ort, status);

    // ensure execution provider is cpu; that's the only one we'll support for now
    if (execution_provider != "cpu") {
        writefln("execution provider '%s' is not supported", execution_provider);
        return -1;
    }

    writefln("using execution provider '%s'", execution_provider);

    writefln("creating inference session for [%s]", mnist_model_path);
    OrtSession* session;
    status = g_ort.CreateSession(env, mnist_model_path.ptr, session_options, &session);
    ort_abort_on_error(g_ort, status);
    writefln("session created");

    // attempt to infer
    //      OrtMemoryInfo* memory_info;
    //   ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    //   const int64_t input_shape[] = {1, 3, 720, 720};
    //   const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
    //   const size_t model_input_len = model_input_ele_count * sizeof(float);

    //   OrtValue* input_tensor = NULL;
    //   ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
    //                                                            input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    //                                                            &input_tensor));
    //   assert(input_tensor != NULL);
    //   int is_tensor;
    //   ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
    //   assert(is_tensor);
    //   g_ort->ReleaseMemoryInfo(memory_info);
    //   const char* input_names[] = {"inputImage"};
    //   const char* output_names[] = {"outputImage"};
    //   OrtValue* output_tensor = NULL;
    //   ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1,
    //                                 &output_tensor));
    //   assert(output_tensor != NULL);
    //   ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
    //   assert(is_tensor);
    //   int ret = 0;
    //   float* output_tensor_data = NULL;
    //   ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
    //   uint8_t* output_image_data = NULL;
    //   chw_to_hwc(output_tensor_data, 720, 720, &output_image_data);
    //   if (write_image_file(output_image_data, 720, 720, output_file) != 0) {
    //     ret = -1;
    //   }
    //   g_ort->ReleaseValue(output_tensor);
    //   g_ort->ReleaseValue(input_tensor);
    //   free(model_input);

    OrtMemoryInfo* memory_info;
    writefln("creating cpu memory info");
    status = g_ort.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_abort_on_error(g_ort, status);

    // mnist input shape is 1,1,28,28
    const long[] input_shape = [1, 1, 28, 28];
    const size_t model_input_len = 1 * 1 * 28 * 28 * float.sizeof;
    float* model_input = cast(float*) malloc(
        input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * float.sizeof);

    OrtValue* input_tensor;
    writefln("creating input tensor");
    status = g_ort.CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape.ptr, input_shape
            .length, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    ort_abort_on_error(g_ort, status);
    assert(input_tensor !is null, "failed to create input tensor");
    int is_tensor;
    status = g_ort.IsTensor(input_tensor, &is_tensor);
    ort_abort_on_error(g_ort, status);

    g_ort.ReleaseMemoryInfo(memory_info);

    char*[] input_names = [cast(char*) "Input3"];
    char*[] output_names = [cast(char*) "Plus214_Output_0"];

    OrtValue* output_tensor;

    writefln("running inference");
    status = g_ort.Run(session, null, input_names.ptr, &input_tensor, 1, output_names.ptr, 1, &output_tensor);
    ort_abort_on_error(g_ort, status);
    assert(output_tensor !is null, "failed to run inference");
    g_ort.IsTensor(output_tensor, &is_tensor);
    assert(is_tensor, "output is not a tensor");

    int ret = 0;
    float* output_tensor_data;
    status = g_ort.GetTensorMutableData(output_tensor, cast(void**)&output_tensor_data);
    ort_abort_on_error(g_ort, status);
    assert(output_tensor_data !is null, "failed to get output tensor data");

    // clean up
    g_ort.ReleaseSessionOptions(session_options);
    g_ort.ReleaseSession(session);
    g_ort.ReleaseEnv(env);

    return 0;
}
