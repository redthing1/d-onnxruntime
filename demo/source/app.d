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

    // clean up
    g_ort.ReleaseSessionOptions(session_options);
    g_ort.ReleaseSession(session);
    g_ort.ReleaseEnv(env);

    return 0;
}
