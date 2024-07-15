module app;

import std.stdio;

import onnxruntime_c_api;

int main() {
    writefln("loading onnxruntime v%s", ORT_API_VERSION);
    const(OrtApi)* g_ort = OrtGetApiBase().GetApi(ORT_API_VERSION);

    return 0;
}
