module onnxruntime_c_api;

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// See docs\c_cxx\README.md on generating the Doxygen documentation from this file

/** \mainpage ONNX Runtime
 *
 * ONNX Runtime is a high-performance inference and training graph execution engine for deep learning models.
 *
 * ONNX Runtime's C, C++ APIs offer an easy to use interface to onboard and execute onnx models.
 * - \subpage c_cpp_api "Core C, C++ APIs"
 * - \subpage training_c_cpp_api "Training C, C++ APIs for on-device training"
 *
 * \page c_cpp_api Core C, C++ APIs
 * <h1>C</h1>
 *
 * ::OrtApi - Click here to go to the structure with all C API functions.
 *
 * <h1>C++</h1>
 *
 * ::Ort - Click here to go to the namespace holding all of the C++ wrapper classes
 *
 * It is a set of header only wrapper classes around the C API. The goal is to turn the C style return value error codes into C++ exceptions, and to
 * automate memory management through standard C++ RAII principles.
 *
 * \addtogroup Global
 * ONNX Runtime C API
 * @{
 */

import core.stdc.config;

extern (C) @nogc nothrow:

/** \brief The API version defined in this header
 *
 * This value is used by some API functions to behave as this version of the header expects.
 */
enum ORT_API_VERSION = 17;

//! @}
// SAL2 Definitions

// Define ORT_DLL_IMPORT if your program is dynamically linked to Ort.
// dllexport is not used, we use a .def file.

// To make symbols visible on macOS/iOS

alias ORTCHAR_T = char;

/// ORTCHAR_T, ORT_TSTR are reserved specifically for path handling.
/// All other strings are UTF-8 encoded, use char and std::string

// When X is a macro, L##X is not defined. In this case, we need to use ORT_TSTR_ON_MACRO.

extern (D) auto ORT_TSTR(T)(auto ref T X) {
    return X;
}

extern (D) auto ORT_TSTR_ON_MACRO(T)(auto ref T X) {
    return X;
}

// On Windows, ORT_FILE is a wchar_t version of the __FILE__ macro.
// Otherwise, ORT_FILE is equivalent to __FILE__.

alias ORT_FILE_INTERNAL = ORT_TSTR;
enum ORT_FILE = ORT_FILE_INTERNAL(__FILE__);

// Any pointer marked with _In_ or _Out_, cannot be NULL.

// Windows users should use unicode paths when possible to bypass the MAX_PATH limitation
// Every pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.
// for ReleaseXXX(...) functions, they can accept NULL pointer.

// For any compiler with C++11 support, MSVC 2015 and greater, or Clang version supporting noexcept.
// Such complex condition is needed because compilers set __cplusplus value differently.

// __VA_ARGS__ on Windows and Linux are different

// XXX: Unfortunately, SAL annotations are known to not work with function pointers

// Used in *.cc files. Almost as same as ORT_API_STATUS, except without ORT_MUST_USE_RESULT and ORT_EXPORT

/** \addtogroup Global
 * ONNX Runtime C API
 * @{
 */

/** Copied from TensorProto::DataType
 * Currently, Ort doesn't support complex64, complex128
 */
enum ONNXTensorElementDataType {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1, // maps to c type float
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2, // maps to c type uint8_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3, // maps to c type int8_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4, // maps to c type uint16_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5, // maps to c type int16_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6, // maps to c type int32_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7, // maps to c type int64_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8, // maps to c++ type std::string
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11, // maps to c type double
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12, // maps to c type uint32_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13, // maps to c type uint64_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14, // complex with float32 real and imaginary components
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15, // complex with float64 real and imaginary components
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16, // Non-IEEE floating-point format based on IEEE754 single-precision
    // float 8 types were introduced in onnx 1.14, see https://onnx.ai/onnx/technical/float8.html
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN = 17, // Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ = 18, // Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 = 19, // Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ = 20 // Non-IEEE floating-point format based on IEEE754 single-precision
}

alias ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2;
alias ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ = ONNXTensorElementDataType
    .ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ;

// Synced with onnx TypeProto oneof
enum ONNXType {
    ONNX_TYPE_UNKNOWN = 0,
    ONNX_TYPE_TENSOR = 1,
    ONNX_TYPE_SEQUENCE = 2,
    ONNX_TYPE_MAP = 3,
    ONNX_TYPE_OPAQUE = 4,
    ONNX_TYPE_SPARSETENSOR = 5,
    ONNX_TYPE_OPTIONAL = 6
}

alias ONNX_TYPE_UNKNOWN = ONNXType.ONNX_TYPE_UNKNOWN;
alias ONNX_TYPE_TENSOR = ONNXType.ONNX_TYPE_TENSOR;
alias ONNX_TYPE_SEQUENCE = ONNXType.ONNX_TYPE_SEQUENCE;
alias ONNX_TYPE_MAP = ONNXType.ONNX_TYPE_MAP;
alias ONNX_TYPE_OPAQUE = ONNXType.ONNX_TYPE_OPAQUE;
alias ONNX_TYPE_SPARSETENSOR = ONNXType.ONNX_TYPE_SPARSETENSOR;
alias ONNX_TYPE_OPTIONAL = ONNXType.ONNX_TYPE_OPTIONAL;

// These types are synced with internal
// SparseFormatFlags
enum OrtSparseFormat {
    ORT_SPARSE_UNDEFINED = 0,
    ORT_SPARSE_COO = 0x1,
    ORT_SPARSE_CSRC = 0x2,
    ORT_SPARSE_BLOCK_SPARSE = 0x4
}

alias ORT_SPARSE_UNDEFINED = OrtSparseFormat.ORT_SPARSE_UNDEFINED;
alias ORT_SPARSE_COO = OrtSparseFormat.ORT_SPARSE_COO;
alias ORT_SPARSE_CSRC = OrtSparseFormat.ORT_SPARSE_CSRC;
alias ORT_SPARSE_BLOCK_SPARSE = OrtSparseFormat.ORT_SPARSE_BLOCK_SPARSE;

// Enum allows to query sparse tensor indices
enum OrtSparseIndicesFormat {
    ORT_SPARSE_COO_INDICES = 0,
    ORT_SPARSE_CSR_INNER_INDICES = 1,
    ORT_SPARSE_CSR_OUTER_INDICES = 2,
    ORT_SPARSE_BLOCK_SPARSE_INDICES = 3
}

alias ORT_SPARSE_COO_INDICES = OrtSparseIndicesFormat.ORT_SPARSE_COO_INDICES;
alias ORT_SPARSE_CSR_INNER_INDICES = OrtSparseIndicesFormat.ORT_SPARSE_CSR_INNER_INDICES;
alias ORT_SPARSE_CSR_OUTER_INDICES = OrtSparseIndicesFormat.ORT_SPARSE_CSR_OUTER_INDICES;
alias ORT_SPARSE_BLOCK_SPARSE_INDICES = OrtSparseIndicesFormat.ORT_SPARSE_BLOCK_SPARSE_INDICES;

/** \brief Logging severity levels
 *
 * In typical API usage, specifying a logging severity level specifies the minimum severity of log messages to show.
 */
enum OrtLoggingLevel {
    ORT_LOGGING_LEVEL_VERBOSE = 0, ///< Verbose informational messages (least severe).
    ORT_LOGGING_LEVEL_INFO = 1, ///< Informational messages.
    ORT_LOGGING_LEVEL_WARNING = 2, ///< Warning messages.
    ORT_LOGGING_LEVEL_ERROR = 3, ///< Error messages.
    ORT_LOGGING_LEVEL_FATAL = 4 ///< Fatal error messages (most severe).
}

alias ORT_LOGGING_LEVEL_VERBOSE = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
alias ORT_LOGGING_LEVEL_INFO = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
alias ORT_LOGGING_LEVEL_WARNING = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
alias ORT_LOGGING_LEVEL_ERROR = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
alias ORT_LOGGING_LEVEL_FATAL = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;

enum OrtErrorCode {
    ORT_OK = 0,
    ORT_FAIL = 1,
    ORT_INVALID_ARGUMENT = 2,
    ORT_NO_SUCHFILE = 3,
    ORT_NO_MODEL = 4,
    ORT_ENGINE_ERROR = 5,
    ORT_RUNTIME_EXCEPTION = 6,
    ORT_INVALID_PROTOBUF = 7,
    ORT_MODEL_LOADED = 8,
    ORT_NOT_IMPLEMENTED = 9,
    ORT_INVALID_GRAPH = 10,
    ORT_EP_FAIL = 11
}

alias ORT_OK = OrtErrorCode.ORT_OK;
alias ORT_FAIL = OrtErrorCode.ORT_FAIL;
alias ORT_INVALID_ARGUMENT = OrtErrorCode.ORT_INVALID_ARGUMENT;
alias ORT_NO_SUCHFILE = OrtErrorCode.ORT_NO_SUCHFILE;
alias ORT_NO_MODEL = OrtErrorCode.ORT_NO_MODEL;
alias ORT_ENGINE_ERROR = OrtErrorCode.ORT_ENGINE_ERROR;
alias ORT_RUNTIME_EXCEPTION = OrtErrorCode.ORT_RUNTIME_EXCEPTION;
alias ORT_INVALID_PROTOBUF = OrtErrorCode.ORT_INVALID_PROTOBUF;
alias ORT_MODEL_LOADED = OrtErrorCode.ORT_MODEL_LOADED;
alias ORT_NOT_IMPLEMENTED = OrtErrorCode.ORT_NOT_IMPLEMENTED;
alias ORT_INVALID_GRAPH = OrtErrorCode.ORT_INVALID_GRAPH;
alias ORT_EP_FAIL = OrtErrorCode.ORT_EP_FAIL;

enum OrtOpAttrType {
    ORT_OP_ATTR_UNDEFINED = 0,
    ORT_OP_ATTR_INT = 1,
    ORT_OP_ATTR_INTS = 2,
    ORT_OP_ATTR_FLOAT = 3,
    ORT_OP_ATTR_FLOATS = 4,
    ORT_OP_ATTR_STRING = 5,
    ORT_OP_ATTR_STRINGS = 6
}

alias ORT_OP_ATTR_UNDEFINED = OrtOpAttrType.ORT_OP_ATTR_UNDEFINED;
alias ORT_OP_ATTR_INT = OrtOpAttrType.ORT_OP_ATTR_INT;
alias ORT_OP_ATTR_INTS = OrtOpAttrType.ORT_OP_ATTR_INTS;
alias ORT_OP_ATTR_FLOAT = OrtOpAttrType.ORT_OP_ATTR_FLOAT;
alias ORT_OP_ATTR_FLOATS = OrtOpAttrType.ORT_OP_ATTR_FLOATS;
alias ORT_OP_ATTR_STRING = OrtOpAttrType.ORT_OP_ATTR_STRING;
alias ORT_OP_ATTR_STRINGS = OrtOpAttrType.ORT_OP_ATTR_STRINGS;

//! @}

/** \addtogroup Global
 * ONNX Runtime C API
 * @{
 */
// The actual types defined have an Ort prefix
struct OrtEnv;
struct OrtStatus; // nullptr for Status* indicates success
struct OrtMemoryInfo;
struct OrtIoBinding;
struct OrtSession; // Don't call ReleaseSession from Dllmain (because session owns a thread pool)
struct OrtValue;
struct OrtRunOptions;
struct OrtTypeInfo;
struct OrtTensorTypeAndShapeInfo;
struct OrtMapTypeInfo;
struct OrtSequenceTypeInfo;
struct OrtOptionalTypeInfo;
struct OrtSessionOptions;
struct OrtCustomOpDomain;
struct OrtModelMetadata;
struct OrtThreadPoolParams;
struct OrtThreadingOptions;
struct OrtArenaCfg;
struct OrtPrepackedWeightsContainer;
struct OrtTensorRTProviderOptionsV2;
struct OrtCUDAProviderOptionsV2;
struct OrtCANNProviderOptions;
struct OrtDnnlProviderOptions;
struct OrtOp;
struct OrtOpAttr;
struct OrtLogger;
struct OrtShapeInferContext;

alias OrtStatusPtr = OrtStatus*;

/** \brief Memory allocation interface
 *
 * Structure of function pointers that defines a memory allocator. This can be created and filled in by the user for custom allocators.
 *
 * When an allocator is passed to any function, be sure that the allocator object is not destroyed until the last allocated object using it is freed.
 */
struct OrtAllocator {
    uint version_; ///< Must be initialized to ORT_API_VERSION
    void* function(OrtAllocator* this_, size_t size) Alloc; ///< Returns a pointer to an allocated block of `size` bytes
    void function(OrtAllocator* this_, void* p) Free; ///< Free a block of memory previously allocated with OrtAllocator::Alloc
    const(OrtMemoryInfo)* function(const(OrtAllocator)* this_) Info; ///< Return a pointer to an ::OrtMemoryInfo that describes this allocator
}

alias OrtLoggingFunction = void function(
    void* param,
    OrtLoggingLevel severity,
    const(char)* category,
    const(char)* logid,
    const(char)* code_location,
    const(char)* message);

/** \brief Graph optimization level
 *
 * Refer to https://www.onnxruntime.ai/docs/performance/graph-optimizations.html#graph-optimization-levels
 * for an in-depth understanding of the Graph Optimization Levels.
 */
enum GraphOptimizationLevel {
    ORT_DISABLE_ALL = 0,
    ORT_ENABLE_BASIC = 1,
    ORT_ENABLE_EXTENDED = 2,
    ORT_ENABLE_ALL = 99
}

alias ORT_DISABLE_ALL = GraphOptimizationLevel.ORT_DISABLE_ALL;
alias ORT_ENABLE_BASIC = GraphOptimizationLevel.ORT_ENABLE_BASIC;
alias ORT_ENABLE_EXTENDED = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
alias ORT_ENABLE_ALL = GraphOptimizationLevel.ORT_ENABLE_ALL;

enum ExecutionMode {
    ORT_SEQUENTIAL = 0,
    ORT_PARALLEL = 1
}

alias ORT_SEQUENTIAL = ExecutionMode.ORT_SEQUENTIAL;
alias ORT_PARALLEL = ExecutionMode.ORT_PARALLEL;

/** \brief Language projection identifiers
 * /see OrtApi::SetLanguageProjection
 */
enum OrtLanguageProjection {
    ORT_PROJECTION_C = 0,
    ORT_PROJECTION_CPLUSPLUS = 1,
    ORT_PROJECTION_CSHARP = 2,
    ORT_PROJECTION_PYTHON = 3,
    ORT_PROJECTION_JAVA = 4,
    ORT_PROJECTION_WINML = 5,
    ORT_PROJECTION_NODEJS = 6
}

alias ORT_PROJECTION_C = OrtLanguageProjection.ORT_PROJECTION_C;
alias ORT_PROJECTION_CPLUSPLUS = OrtLanguageProjection.ORT_PROJECTION_CPLUSPLUS;
alias ORT_PROJECTION_CSHARP = OrtLanguageProjection.ORT_PROJECTION_CSHARP;
alias ORT_PROJECTION_PYTHON = OrtLanguageProjection.ORT_PROJECTION_PYTHON;
alias ORT_PROJECTION_JAVA = OrtLanguageProjection.ORT_PROJECTION_JAVA;
alias ORT_PROJECTION_WINML = OrtLanguageProjection.ORT_PROJECTION_WINML;
alias ORT_PROJECTION_NODEJS = OrtLanguageProjection.ORT_PROJECTION_NODEJS;

struct OrtKernelInfo;
struct OrtKernelContext;

enum OrtAllocatorType {
    OrtInvalidAllocator = -1,
    OrtDeviceAllocator = 0,
    OrtArenaAllocator = 1
}

alias OrtInvalidAllocator = OrtAllocatorType.OrtInvalidAllocator;
alias OrtDeviceAllocator = OrtAllocatorType.OrtDeviceAllocator;
alias OrtArenaAllocator = OrtAllocatorType.OrtArenaAllocator;

/** \brief Memory types for allocated memory, execution provider specific types should be extended in each provider.
 */
// Whenever this struct is updated, please also update the MakeKey function in onnxruntime / core / framework / execution_provider.cc
enum OrtMemType {
    OrtMemTypeCPUInput = -2, ///< Any CPU memory used by non-CPU execution provider
    OrtMemTypeCPUOutput = -1, ///< CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
    OrtMemTypeCPU = OrtMemTypeCPUOutput, ///< Temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
    OrtMemTypeDefault = 0 ///< The default allocator for execution provider
}

alias OrtMemTypeCPUInput = OrtMemType.OrtMemTypeCPUInput;
alias OrtMemTypeCPUOutput = OrtMemType.OrtMemTypeCPUOutput;
alias OrtMemTypeCPU = OrtMemType.OrtMemTypeCPU;
alias OrtMemTypeDefault = OrtMemType.OrtMemTypeDefault;

/** \brief This mimics OrtDevice type constants so they can be returned in the API
 */
enum OrtMemoryInfoDeviceType {
    OrtMemoryInfoDeviceType_CPU = 0,
    OrtMemoryInfoDeviceType_GPU = 1,
    OrtMemoryInfoDeviceType_FPGA = 2
}

alias OrtMemoryInfoDeviceType_CPU = OrtMemoryInfoDeviceType.OrtMemoryInfoDeviceType_CPU;
alias OrtMemoryInfoDeviceType_GPU = OrtMemoryInfoDeviceType.OrtMemoryInfoDeviceType_GPU;
alias OrtMemoryInfoDeviceType_FPGA = OrtMemoryInfoDeviceType.OrtMemoryInfoDeviceType_FPGA;

/** \brief Algorithm to use for cuDNN Convolution Op
 */
enum OrtCudnnConvAlgoSearch {
    OrtCudnnConvAlgoSearchExhaustive = 0, // expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
    OrtCudnnConvAlgoSearchHeuristic = 1, // lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
    OrtCudnnConvAlgoSearchDefault = 2 // default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
}

alias OrtCudnnConvAlgoSearchExhaustive = OrtCudnnConvAlgoSearch.OrtCudnnConvAlgoSearchExhaustive;
alias OrtCudnnConvAlgoSearchHeuristic = OrtCudnnConvAlgoSearch.OrtCudnnConvAlgoSearchHeuristic;
alias OrtCudnnConvAlgoSearchDefault = OrtCudnnConvAlgoSearch.OrtCudnnConvAlgoSearchDefault;

/** \brief CUDA Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_CUDA
 */
struct OrtCUDAProviderOptions {
    /** \brief CUDA device Id
     *   Defaults to 0.
     */
    int device_id;

    /** \brief CUDA Convolution algorithm search configuration.
     *   See enum OrtCudnnConvAlgoSearch for more details.
     *   Defaults to OrtCudnnConvAlgoSearchExhaustive.
     */
    OrtCudnnConvAlgoSearch cudnn_conv_algo_search;

    /** \brief CUDA memory limit (To use all possible memory pass in maximum size_t)
     *   Defaults to SIZE_MAX.
     *   \note If a ::OrtArenaCfg has been applied, it will override this field
     */
    alias size_t = c_ulong;
    size_t gpu_mem_limit;

    /** \brief Strategy used to grow the memory arena
     *   0 = kNextPowerOfTwo<br>
     *   1 = kSameAsRequested<br>
     *   Defaults to 0.
     *   \note If a ::OrtArenaCfg has been applied, it will override this field
     */
    int arena_extend_strategy;

    /** \brief Flag indicating if copying needs to take place on the same stream as the compute stream in the CUDA EP
     *   0 = Use separate streams for copying and compute.
     *   1 = Use the same stream for copying and compute.
     *   Defaults to 1.
     *   WARNING: Setting this to 0 may result in data races for some models.
     *   Please see issue #4829 for more details.
     */
    int do_copy_in_default_stream;

    /** \brief Flag indicating if there is a user provided compute stream
     *   Defaults to 0.
     */
    int has_user_compute_stream;

    /** \brief User provided compute stream.
     *   If provided, please set `has_user_compute_stream` to 1.
     */
    void* user_compute_stream;

    /** \brief CUDA memory arena configuration parameters
     */
    OrtArenaCfg* default_memory_arena_cfg;

    /** \brief Enable TunableOp for using.
     *   Set it to 1/0 to enable/disable TunableOp. Otherwise, it is disabled by default.
     *   This option can be overriden by environment variable ORT_CUDA_TUNABLE_OP_ENABLE.
     */
    int tunable_op_enable;

    /** \brief Enable TunableOp for tuning.
     *   Set it to 1/0 to enable/disable TunableOp tuning. Otherwise, it is disabled by default.
     *   This option can be overriden by environment variable ORT_CUDA_TUNABLE_OP_TUNING_ENABLE.
     */
    int tunable_op_tuning_enable;

    /** \brief Max tuning duration time limit for each instance of TunableOp.
     *   Defaults to 0 to disable the limit.
     */
    int tunable_op_max_tuning_duration_ms;
}

/** \brief ROCM Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_ROCM
 */
struct OrtROCMProviderOptions {
    /** \brief ROCM device Id
     *   Defaults to 0.
     */
    int device_id;

    /** \brief ROCM MIOpen Convolution algorithm exaustive search option.
     *   Defaults to 0 (false).
     */
    int miopen_conv_exhaustive_search;

    /** \brief ROCM memory limit (To use all possible memory pass in maximum size_t)
     *   Defaults to SIZE_MAX.
     *   \note If a ::OrtArenaCfg has been applied, it will override this field
     */
    size_t gpu_mem_limit;

    /** \brief Strategy used to grow the memory arena
     *   0 = kNextPowerOfTwo<br>
     *   1 = kSameAsRequested<br>
     *   Defaults to 0.
     *   \note If a ::OrtArenaCfg has been applied, it will override this field
     */
    int arena_extend_strategy;

    /** \brief Flag indicating if copying needs to take place on the same stream as the compute stream in the ROCM EP
     *   0 = Use separate streams for copying and compute.
     *   1 = Use the same stream for copying and compute.
     *   Defaults to 1.
     *   WARNING: Setting this to 0 may result in data races for some models.
     *   Please see issue #4829 for more details.
     */
    int do_copy_in_default_stream;

    /** \brief Flag indicating if there is a user provided compute stream
     *   Defaults to 0.
     */
    int has_user_compute_stream;

    /** \brief User provided compute stream.
     *   If provided, please set `has_user_compute_stream` to 1.
     */
    void* user_compute_stream;

    /** \brief ROCM memory arena configuration parameters
     */
    OrtArenaCfg* default_memory_arena_cfg;

    /** \brief Enable TunableOp for using.
     *   Set it to 1/0 to enable/disable TunableOp. Otherwise, it is disabled by default.
     *   This option can be overriden by environment variable ORT_ROCM_TUNABLE_OP_ENABLE.
     */
    int tunable_op_enable;

    /** \brief Enable TunableOp for tuning.
     *   Set it to 1/0 to enable/disable TunableOp tuning. Otherwise, it is disabled by default.
     *   This option can be overriden by environment variable ORT_ROCM_TUNABLE_OP_TUNING_ENABLE.
     */
    int tunable_op_tuning_enable;

    /** \brief Max tuning duration time limit for each instance of TunableOp.
     *   Defaults to 0 to disable the limit.
     */
    int tunable_op_max_tuning_duration_ms;
}

/** \brief TensorRT Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_TensorRT
 */
struct OrtTensorRTProviderOptions {
    int device_id; ///< CUDA device id (0 = default device)
    int has_user_compute_stream; // indicator of user specified CUDA compute stream.
    void* user_compute_stream; // user specified CUDA compute stream.
    int trt_max_partition_iterations; // maximum iterations for TensorRT parser to get capability
    int trt_min_subgraph_size; // minimum size of TensorRT subgraphs
    size_t trt_max_workspace_size; // maximum workspace size for TensorRT.
    int trt_fp16_enable; // enable TensorRT FP16 precision. Default 0 = false, nonzero = true
    int trt_int8_enable; // enable TensorRT INT8 precision. Default 0 = false, nonzero = true
    const(char)* trt_int8_calibration_table_name; // TensorRT INT8 calibration table name.
    int trt_int8_use_native_calibration_table; // use native TensorRT generated calibration table. Default 0 = false, nonzero = true
    int trt_dla_enable; // enable DLA. Default 0 = false, nonzero = true
    int trt_dla_core; // DLA core number. Default 0
    int trt_dump_subgraphs; // dump TRT subgraph. Default 0 = false, nonzero = true
    int trt_engine_cache_enable; // enable engine caching. Default 0 = false, nonzero = true
    const(char)* trt_engine_cache_path; // specify engine cache path
    int trt_engine_decryption_enable; // enable engine decryption. Default 0 = false, nonzero = true
    const(char)* trt_engine_decryption_lib_path; // specify engine decryption library path
    int trt_force_sequential_engine_build; // force building TensorRT engine sequentially. Default 0 = false, nonzero = true
    // This is the legacy struct and don't add new fields here.
    // For new field that can be represented by string, please add it in include/onnxruntime/core/providers/tensorrt/tensorrt_provider_options.h
    // For non-string field, need to create a new separate api to handle it.
}

/** \brief MIGraphX Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_MIGraphX
 */
struct OrtMIGraphXProviderOptions {
    int device_id; // hip device id.
    int migraphx_fp16_enable; // MIGraphX FP16 precision. Default 0 = false, nonzero = true
    int migraphx_int8_enable; // MIGraphX INT8 precision. Default 0 = false, nonzero = true
    int migraphx_use_native_calibration_table; // MIGraphx INT8 cal table. Default 0 = false, noznero = true
    const(char)* migraphx_int8_calibration_table_name; // MIGraphx INT8 calibration table name
}

/** \brief OpenVINO Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_OpenVINO
 */
struct OrtOpenVINOProviderOptions {
    /** \brief Device type string
     *
     * Valid settings are one of: "CPU_FP32", "CPU_FP16", "GPU_FP32", "GPU_FP16"
     */
    const(char)* device_type;
    ubyte enable_npu_fast_compile; ///< 0 = disabled, nonzero = enabled
    const(char)* device_id;
    size_t num_of_threads; ///< 0 = Use default number of threads
    const(char)* cache_dir; // path is set to empty by default
    void* context;
    ubyte enable_opencl_throttling; ///< 0 = disabled, nonzero = enabled
    ubyte enable_dynamic_shapes; ///< 0 = disabled, nonzero = enabled
}

struct OrtTrainingApi;

/** \brief The helper interface to get the right version of OrtApi
 *
 * Get a pointer to this structure through ::OrtGetApiBase
 */
struct OrtApiBase {
    /** \brief Get a pointer to the requested version of the ::OrtApi
     *
     * \param[in] version Must be ::ORT_API_VERSION
     * \return The ::OrtApi for the version requested, nullptr will be returned if this version is unsupported, for example when using a runtime
     *   older than the version created with this header file.
     *
     * One can call GetVersionString() to get the version of the Onnxruntime library for logging
     * and error reporting purposes.
     */
    const(OrtApi)* function(uint version_) GetApi;

    /** \brief Returns a null terminated string of the version of the Onnxruntime library (eg: "1.8.1")
     *
     *  \return UTF-8 encoded version string. Do not deallocate the returned buffer.
     */
    const(char)* function() GetVersionString;
}

/** \brief The Onnxruntime library's entry point to access the C API
 *
 * Call this to get the a pointer to an ::OrtApiBase
 */
const(OrtApiBase)* OrtGetApiBase();

/** \brief Thread work loop function
 *
 * Onnxruntime will provide the working loop on custom thread creation
 * Argument is an onnxruntime built-in type which will be provided when thread pool calls OrtCustomCreateThreadFn
 */
alias OrtThreadWorkerFn = void function(void* ort_worker_fn_param);

struct OrtCustomHandleType {
    char __place_holder;
}

alias OrtCustomThreadHandle = const(OrtCustomHandleType)*;

/** \brief Ort custom thread creation function
 *
 * The function should return a thread handle to be used in onnxruntime thread pools
 * Onnxruntime will throw exception on return value of nullptr or 0, indicating that the function failed to create a thread
 */
alias OrtCustomCreateThreadFn = const(OrtCustomHandleType)* function(void* ort_custom_thread_creation_options, OrtThreadWorkerFn ort_thread_worker_fn, void* ort_worker_fn_param);

/** \brief Custom thread join function
 *
 * Onnxruntime thread pool destructor will call the function to join a custom thread.
 * Argument ort_custom_thread_handle is the value returned by OrtCustomCreateThreadFn
 */
alias OrtCustomJoinThreadFn = void function(OrtCustomThreadHandle ort_custom_thread_handle);

alias RegisterCustomOpsFn = OrtStatus* function(OrtSessionOptions* options, const(OrtApiBase)* api);

/** \brief Callback function for RunAsync
 *
 * \param[in] user_data User specific data that passed back to the callback
 * \param[out] outputs On succeed, outputs host inference results, on error, the value will be nullptr
 * \param[out] num_outputs Number of outputs, on error, the value will be zero
 * \param[out] status On error, status will provide details
 */
alias RunAsyncCallbackFn = void function(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status);

/** \brief The C API
 *
 * All C API functions are defined inside this structure as pointers to functions.
 * Call OrtApiBase::GetApi to get a pointer to it
 *
 * \nosubgrouping
 */
struct OrtApi {
    /// \name OrtStatus
    /// @{

    /**
     * \brief Create an OrtStatus from a null terminated string
     *
     * \param[in] code
     * \param[in] msg A null-terminated string. Its contents will be copied.
     * \return A new OrtStatus object, must be destroyed with OrtApi::ReleaseStatus
     */
    OrtStatus* function(OrtErrorCode code, const(char)* msg) CreateStatus;

    /** \brief Get OrtErrorCode from OrtStatus
     *
     * \param[in] status
     * \return OrtErrorCode that \p status was created with
     */
    OrtErrorCode function(const(OrtStatus)* status) GetErrorCode;

    /** \brief Get error string from OrtStatus
     *
     * \param[in] status
     * \return The error message inside the `status`. Do not free the returned value.
     */
    const(char)* function(const(OrtStatus)* status) GetErrorMessage;

    /// @}
    /// \name OrtEnv
    /// @{

    /** \brief Create an OrtEnv
     *
     * \note Invoking this function will return the same instance of the environment as that returned by a previous call
     * to another env creation function; all arguments to this function will be ignored.
     * \param[in] log_severity_level The log severity level.
     * \param[in] logid The log identifier.
     * \param[out] out Returned newly created OrtEnv. Must be freed with OrtApi::ReleaseEnv
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtLoggingLevel log_severity_level, const(char)* logid, OrtEnv** out_) CreateEnv;

    /** \brief Create an OrtEnv
     *
     * \note Invoking this function will return the same instance of the environment as that returned by a previous call
     * to another env creation function; all arguments to this function will be ignored. If you want to provide your
     * own logging function, consider setting it using the SetUserLoggingFunction API instead.
     * \param[in] logging_function A pointer to a logging function.
     * \param[in] logger_param A pointer to arbitrary data passed as the ::OrtLoggingFunction `param` parameter to
     *                         `logging_function`. This parameter is optional.
     * \param[in] log_severity_level The log severity level.
     * \param[in] logid The log identifier.
     * \param[out] out Returned newly created OrtEnv. Must be freed with OrtApi::ReleaseEnv
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtLoggingFunction logging_function,
        void* logger_param,
        OrtLoggingLevel log_severity_level,
        const(char)* logid,
        OrtEnv** out_) CreateEnvWithCustomLogger;

    /** \brief Enable Telemetry
     *
     * \note Telemetry events are on by default since they are lightweight
     * \param[in] env
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtEnv)* env) EnableTelemetryEvents;
    /** \brief Disable Telemetry
     *
     * \see OrtApi::EnableTelemetryEvents
     * \param[in] env
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtEnv)* env) DisableTelemetryEvents;

    /// @}
    /// \name OrtSession
    /// @{

    /** \brief Create an OrtSession from a model file
     *
     * \param[in] env
     * \param[in] model_path
     * \param[in] options
     * \param[out] out Returned newly created OrtSession. Must be freed with OrtApi::ReleaseSession
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    // TODO: document the path separator convention? '/' vs '\'
    // TODO: should specify the access characteristics of model_path. Is this read only during the
    // execution of CreateSession, or does the OrtSession retain a handle to the file/directory
    // and continue to access throughout the OrtSession lifetime?
    //  What sort of access is needed to model_path : read or read/write?
    OrtStatusPtr function(
        const(OrtEnv)* env,
        const(char)* model_path,
        const(OrtSessionOptions)* options,
        OrtSession** out_) CreateSession;

    /** \brief Create an OrtSession from memory
     *
     * \param[in] env
     * \param[in] model_data
     * \param[in] model_data_length
     * \param[in] options
     * \param[out] out Returned newly created OrtSession. Must be freed with OrtApi::ReleaseSession
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtEnv)* env,
        const(void)* model_data,
        size_t model_data_length,
        const(OrtSessionOptions)* options,
        OrtSession** out_) CreateSessionFromArray;

    /** \brief Run the model in an ::OrtSession
     *
     * Will not return until the model run has completed. Multiple threads might be used to run the model based on
     * the options in the ::OrtSession and settings used when creating the ::OrtEnv
     *
     * \param[in] session
     * \param[in] run_options If nullptr, will use a default ::OrtRunOptions
     * \param[in] input_names Array of null terminated UTF8 encoded strings of the input names
     * \param[in] inputs Array of ::OrtValue%s of the input values
     * \param[in] input_len Number of elements in the input_names and inputs arrays
     * \param[in] output_names Array of null terminated UTF8 encoded strings of the output names
     * \param[in] output_names_len Number of elements in the output_names and outputs array
     * \param[out] outputs Array of ::OrtValue%s that the outputs are stored in. This can also be
     *     an array of nullptr values, in this case ::OrtValue objects will be allocated and pointers
     *     to them will be set into the `outputs` array.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSession* session,
        const(OrtRunOptions)* run_options,
        const(char*)* input_names,
        const(OrtValue*)* inputs,
        size_t input_len,
        const(char*)* output_names,
        size_t output_names_len,
        OrtValue** outputs) Run;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Create an ::OrtSessionOptions object
     *
     * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
     * functions to enable them in the session:<br>
     *   OrtSessionOptionsAppendExecutionProvider_CPU<br>
     *   OrtSessionOptionsAppendExecutionProvider_CUDA<br>
     *   OrtSessionOptionsAppendExecutionProvider_(remaining providers...)<br>
     * The order they are called indicates the preference order as well. In other words call this method
     * on your most preferred execution provider first followed by the less preferred ones.
     * If none are called Ort will use its internal CPU execution provider.
     *
     * \param[out] options The newly created OrtSessionOptions. Must be freed with OrtApi::ReleaseSessionOptions
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions** options) CreateSessionOptions;

    /** \brief Set filepath to save optimized model after graph level transformations
     *
     * \param[in] options
     * \param[in] optimized_model_filepath
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(char)* optimized_model_filepath) SetOptimizedModelFilePath;

    /** \brief Create a copy of an existing ::OrtSessionOptions
     *
     * \param[in] in_options OrtSessionOptions to copy
     * \param[out] out_options Returned newly created ::OrtSessionOptions. Must be freed with OrtApi::ReleaseSessionOptions
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtSessionOptions)* in_options,
        OrtSessionOptions** out_options) CloneSessionOptions;

    /** \brief Set execution mode
     *
     * Controls whether you want to execute operators in your graph sequentially or in parallel. Usually when the model
     *  has many branches, setting this option to ExecutionMode.ORT_PARALLEL will give you better performance.
     *  See [docs/ONNX_Runtime_Perf_Tuning.md] for more details.
     *
     * \param[in] options
     * \param[in] execution_mode
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, ExecutionMode execution_mode) SetSessionExecutionMode;

    /** \brief Enable profiling for a session
     *
     * \param[in] options
     * \param[in] profile_file_prefix
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, const(char)* profile_file_prefix) EnableProfiling;

    /** \brief Disable profiling for a session
     *
     * \param[in] options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options) DisableProfiling;

    /** \brief Enable the memory pattern optimization
     *
     * The idea is if the input shapes are the same, we could trace the internal memory allocation
     * and generate a memory pattern for future request. So next time we could just do one allocation
     * with a big chunk for all the internal memory allocation.
     * \note Memory pattern optimization is only available when Sequential Execution mode is enabled (see OrtApi::SetSessionExecutionMode)
     *
     * \see OrtApi::DisableMemPattern
     *
     * \param[in] options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options) EnableMemPattern;

    /** \brief Disable the memory pattern optimization
     *
     * \see OrtApi::EnableMemPattern
     *
     * \param[in] options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options) DisableMemPattern;

    /** \brief Enable the memory arena on CPU
     *
     * Arena may pre-allocate memory for future usage.
     *
     * \param[in] options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options) EnableCpuMemArena;

    /** \brief Disable the memory arena on CPU
     *
     * \param[in] options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options) DisableCpuMemArena;

    /** \brief Set session log id
     *
     * \param[in] options
     * \param[in] logid The log identifier.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, const(char)* logid) SetSessionLogId;

    /** \brief Set session log verbosity level
     *
     * Applies to session load, initialization, etc
     *
     * \param[in] options
     * \param[in] session_log_verbosity_level \snippet{doc} snippets.dox Log Verbosity Level
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, int session_log_verbosity_level) SetSessionLogVerbosityLevel;

    /** \brief Set session log severity level
     *
     * \param[in] options
     * \param[in] session_log_severity_level The log severity level (refer to ::OrtLoggingLevel for possible values).
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, int session_log_severity_level) SetSessionLogSeverityLevel;

    /** \brief Set the optimization level to apply when loading a graph
     *
     * Please see https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html for an in-depth explanation
     * \param[in,out] options The session options object
     * \param[in] graph_optimization_level The optimization level
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        GraphOptimizationLevel graph_optimization_level) SetSessionGraphOptimizationLevel;

    /** \brief Sets the number of threads used to parallelize the execution within nodes
     *
     * When running a single node operation, ex. add, this sets the maximum number of threads to use.
     *
     * \note If built with OpenMP, this has no effect on the number of threads used. In this case
     *       use the OpenMP env variables to configure the number of intra op num threads.
     *
     * \param[in] options
     * \param[in] intra_op_num_threads Number of threads to use<br>
     *   A value of 0 will use the default number of threads<br>
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, int intra_op_num_threads) SetIntraOpNumThreads;

    /** \brief Sets the number of threads used to parallelize the execution of the graph
     *
     * If nodes can be run in parallel, this sets the maximum number of threads to use to run them in parallel.
     *
     * \note If sequential execution is enabled this value is ignored, it acts as if it was set to 1.
     *
     * \param[in] options
     * \param[in] inter_op_num_threads Number of threads to use<br>
     *   A value of 0 will use the default number of threads<br>
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, int inter_op_num_threads) SetInterOpNumThreads;

    /// @}
    /// \name OrtCustomOpDomain
    /// @{

    /** \brief Create a custom op domain
     *
     * \param[in] domain
     * \param[out] out Newly created domain. Must be freed with OrtApi::ReleaseCustomOpDomain
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(char)* domain, OrtCustomOpDomain** out_) CreateCustomOpDomain;

    /** \brief Add a custom op to a custom op domain
     *
     * \note The OrtCustomOp* pointer must remain valid until the ::OrtCustomOpDomain using it is released
     *
     * \param[in] custom_op_domain
     * \param[in] op
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtCustomOpDomain* custom_op_domain, const(OrtCustomOp)* op) CustomOpDomain_Add;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Add custom op domain to a session options
     *
     * \note The OrtCustomOpDomain* must not be deleted until all sessions using it are released
     *
     * \param[in] options
     * \param[in] custom_op_domain
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, OrtCustomOpDomain* custom_op_domain) AddCustomOpDomain;

    /** \deprecated Use OrtApi::RegisterCustomOpsLibrary_V2.
     *
     * Registers custom ops from a shared library.
     *
     * Loads a shared library (dll on windows, so on linux, etc) named 'library_path' and looks for this entry point:
     *		OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
     * It then passes in the provided session options to this function along with the api base.
     * The handle to the loaded library is returned in library_handle. It can be freed by the caller after all sessions using the passed in
     * session options are destroyed, or if an error occurs and it is non null.
     *
     * \param[in] options
     * \param[in] library_path
     * \param[out] library_handle OS specific handle to the loaded library (Use FreeLibrary on Windows, dlclose on Linux, etc.. to unload)
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, const(char)* library_path, void** library_handle) RegisterCustomOpsLibrary;

    /// @}
    /// \name OrtSession
    /// @{

    /** \brief Get input count for a session
     *
     * This number must also match the number of inputs passed to OrtApi::Run
     *
     * \see OrtApi::SessionGetInputTypeInfo, OrtApi::SessionGetInputName, OrtApi::Session
     *
     * \param[in] session
     * \param[out] out Number of inputs
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, size_t* out_) SessionGetInputCount;

    /** \brief Get output count for a session
     *
     * This number must also match the number of outputs returned by OrtApi::Run
     *
     * \see OrtApi::SessionGetOutputTypeInfo, OrtApi::SessionGetOutputName, OrtApi::Session
     *
     * \param[in] session
     * \param[out] out Number of outputs
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, size_t* out_) SessionGetOutputCount;

    /** \brief Get overridable initializer count
     *
     * \see OrtApi::SessionGetOverridableInitializerTypeInfo, OrtApi::SessionGetOverridableInitializerName
     *
     * \param[in] session
     * \param[in] out
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, size_t* out_) SessionGetOverridableInitializerCount;

    /** \brief Get input type information
     *
     * \param[in] session
     * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetInputCount returns (exclusive)
     * \param[out] type_info Must be freed with OrtApi::ReleaseTypeInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, size_t index, OrtTypeInfo** type_info) SessionGetInputTypeInfo;

    /** \brief Get output type information
     *
     * \param[in] session
     * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOutputCount returns (exclusive)
     * \param[out] type_info Must be freed with OrtApi::ReleaseTypeInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, size_t index, OrtTypeInfo** type_info) SessionGetOutputTypeInfo;

    /** \brief Get overridable initializer type information
     *
     * \param[in] session
     * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOverridableInitializerCount returns (exclusive)
     * \param[out] type_info Must be freed with OrtApi::ReleaseTypeInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, size_t index, OrtTypeInfo** type_info) SessionGetOverridableInitializerTypeInfo;

    /** \brief Get input name
     *
     * \param[in] session
     * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetInputCount returns (exclusive)
     * \param[in] allocator
     * \param[out] value Set to a null terminated UTF-8 encoded string allocated using `allocator`. Must be freed using `allocator`.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, size_t index, OrtAllocator* allocator, char** value) SessionGetInputName;

    /** \brief Get output name
     *
     * \param[in] session
     * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOutputCount returns (exclusive)
     * \param[in] allocator
     * \param[out] value Set to a null terminated UTF-8 encoded string allocated using `allocator`. Must be freed using `allocator`.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, size_t index, OrtAllocator* allocator, char** value) SessionGetOutputName;

    /** \brief Get overridable initializer name
     *
     * \param[in] session
     * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOverridableInitializerCount returns (exclusive)
     * \param[in] allocator
     * \param[out] value Set to a null terminated UTF-8 encoded string allocated using `allocator`. Must be freed using `allocator`.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtSession)* session,
        size_t index,
        OrtAllocator* allocator,
        char** value) SessionGetOverridableInitializerName;

    /// @}
    /// \name OrtRunOptions
    /// @{

    /** \brief Create an OrtRunOptions
     *
     * \param[out] out Returned newly created ::OrtRunOptions. Must be freed with OrtApi::ReleaseRunOptions
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtRunOptions** out_) CreateRunOptions;

    /** \brief Set per-run log verbosity level
     *
     * \see OrtApi::RunOptionsGetRunLogVerbosityLevel
     *
     * \param[in] options
     * \param[in] log_verbosity_level \snippet{doc} snippets.dox Log Verbosity Level
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtRunOptions* options, int log_verbosity_level) RunOptionsSetRunLogVerbosityLevel;

    /** \brief Set per-run log severity level
     *
     * \see OrtApi::RunOptionsGetRunLogSeverityLevel
     *
     * \param[in] options
     * \param[in] log_severity_level The log severity level (refer to ::OrtLoggingLevel for possible values).
     */
    OrtStatusPtr function(OrtRunOptions* options, int log_severity_level) RunOptionsSetRunLogSeverityLevel;

    /** \brief Set per-run tag
     *
     * This is used in a per-run log identifier.
     *
     * \see OrtApi::RunOptionsGetRunTag
     *
     * \param[in] options
     * \param[in] run_tag The run tag.
     */
    OrtStatusPtr function(OrtRunOptions* options, const(char)* run_tag) RunOptionsSetRunTag;

    /** \brief Get per-run log verbosity level
     *
     * \see OrtApi::RunOptionsSetRunLogVerbosityLevel
     *
     * \param[in] options
     * \param[out] log_verbosity_level \snippet{doc} snippets.dox Log Verbosity Level
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtRunOptions)* options,
        int* log_verbosity_level) RunOptionsGetRunLogVerbosityLevel;

    /** \brief Get per-run log severity level
     *
     * \see OrtApi::RunOptionsSetRunLogSeverityLevel
     *
     * \param[in] options
     * \param[out] log_severity_level The log severity level (refer to ::OrtLoggingLevel for possible values).
     */
    OrtStatusPtr function(const(OrtRunOptions)* options, int* log_severity_level) RunOptionsGetRunLogSeverityLevel;

    /** \brief Get per-run tag
     *
     * This is used in a per-run log identifier.
     *
     * \see OrtApi::RunOptionsSetRunTag
     *
     * \param[in] options
     * \param[out] run_tag The run tag.
     *                     Do not free this value, it is owned by `options`. It will be invalidated if the run tag
     *                     changes (i.e., with OrtApi::RunOptionsSetRunTag) or `options` is freed.
     */
    OrtStatusPtr function(const(OrtRunOptions)* options, const(char*)* run_tag) RunOptionsGetRunTag;

    /** \brief Set terminate flag
     *
     * If a currently executing session needs to be force terminated, this can be called from another thread to force it to fail with an error.
     *
     * \param[in] options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtRunOptions* options) RunOptionsSetTerminate;

    /** \brief Clears the terminate flag
     *
     * Used so the OrtRunOptions instance can be used in a new OrtApi::Run call without it instantly terminating
     *
     * \param[in] options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtRunOptions* options) RunOptionsUnsetTerminate;

    /// @}
    /// \name OrtValue
    /// @{

    /** \brief Create a tensor
     *
     * Create a tensor using a supplied ::OrtAllocator
     *
     * \param[in] allocator
     * \param[in] shape Pointer to the tensor shape dimensions.
     * \param[in] shape_len The number of tensor shape dimensions.
     * \param[in] type
     * \param[out] out Returns newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtAllocator* allocator,
        const(long)* shape,
        size_t shape_len,
        ONNXTensorElementDataType type,
        OrtValue** out_) CreateTensorAsOrtValue;

    /** \brief Create a tensor backed by a user supplied buffer
     *
     * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
     * p_data is owned by caller. ReleaseValue won't release p_data.
     *
     * \param[in] info Memory description of where the p_data buffer resides (CPU vs GPU etc).
     * \param[in] p_data Pointer to the data buffer.
     * \param[in] p_data_len The number of bytes in the data buffer.
     * \param[in] shape Pointer to the tensor shape dimensions.
     * \param[in] shape_len The number of tensor shape dimensions.
     * \param[in] type The data type.
     * \param[out] out Returns newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtMemoryInfo)* info,
        void* p_data,
        size_t p_data_len,
        const(long)* shape,
        size_t shape_len,
        ONNXTensorElementDataType type,
        OrtValue** out_) CreateTensorWithDataAsOrtValue;

    /** \brief Return if an ::OrtValue is a tensor type
     *
     * \param[in] value A tensor type (string tensors are not supported)
     * \param[out] out Set to 1 iff ::OrtValue is a tensor, 0 otherwise
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, int* out_) IsTensor;

    /** \brief Get a pointer to the raw data inside a tensor
     *
     * Used to read/write/modify the internal tensor data directly.
     * \note The returned pointer is valid until the \p value is destroyed.
     *
     * \param[in] value A tensor type (string tensors are not supported)
     * \param[out] out Filled in with a pointer to the internal storage
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtValue* value, void** out_) GetTensorMutableData;

    /** \brief Set all strings at once in a string tensor
     *
     * \param[in,out] value A tensor of type ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
     * \param[in] s An array of strings. Each string in this array must be null terminated.
     * \param[in] s_len Count of strings in s (Must match the size of \p value's tensor shape)
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtValue* value, const(char*)* s, size_t s_len) FillStringTensor;

    /** \brief Get total byte length for all strings in a string tensor
     *
     * Typically used with OrtApi::GetStringTensorContent
     *
     * \param[in] value A tensor of type ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
     * \param[out] len Total byte length of all strings (does not include trailing nulls)
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, size_t* len) GetStringTensorDataLength;

    /** \brief Get all strings from a string tensor
     *
     * An example of the results:<br>
     * Given \p value is a string tensor with the strings { "This" "is" "a" "test" }<br>
     * \p s must have a size of 11 bytes<br>
     * \p offsets must have 4 elements<br>
     * After the call, these values will be filled in:<br>
     * \p s will contain "Thisisatest"<br>
     * \p offsets will contain { 0, 4, 6, 7 }<br>
     * The length of the last string is just s_len - offsets[last]
     *
     * \param[in] value A tensor of type ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
     * \param[in] s Buffer to sequentially write all tensor strings to. Each string is NOT null-terminated.
     * \param[in] s_len Number of bytes of buffer pointed to by \p s (Get it from OrtApi::GetStringTensorDataLength)
     * \param[out] offsets Array of start offsets into the strings written to \p s
     * \param[in] offsets_len Number of elements in offsets
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtValue)* value,
        void* s,
        size_t s_len,
        size_t* offsets,
        size_t offsets_len) GetStringTensorContent;

    /// @}
    /// \name OrtTypeInfo
    /// @{

    /** \brief Get ::OrtTensorTypeAndShapeInfo from an ::OrtTypeInfo
     *
     * \param[in] type_info
     * \param[out] out Do not free this value, it will be valid until type_info is freed.
     *             If type_info does not represent tensor, this value will be set to nullptr.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtTypeInfo)* type_info,
        const(OrtTensorTypeAndShapeInfo*)* out_) CastTypeInfoToTensorInfo;

    /** \brief Get ::ONNXType from ::OrtTypeInfo
     *
     * \param[in] type_info
     * \param[out] out
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtTypeInfo)* type_info, ONNXType* out_) GetOnnxTypeFromTypeInfo;

    /// @}
    /// \name OrtTensorTypeAndShapeInfo
    /// @{

    /** \brief Create an ::OrtTensorTypeAndShapeInfo object
     *
     * \param[out] out Returns newly created ::OrtTensorTypeAndShapeInfo. Must be freed with OrtApi::ReleaseTensorTypeAndShapeInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtTensorTypeAndShapeInfo** out_) CreateTensorTypeAndShapeInfo;

    /** \brief Set element type in ::OrtTensorTypeAndShapeInfo
     *
     * \param[in] info
     * \param[in] type
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtTensorTypeAndShapeInfo* info, ONNXTensorElementDataType type) SetTensorElementType;

    /** \brief Set shape information in ::OrtTensorTypeAndShapeInfo
     *
     * \param[in] info
     * \param[in] dim_values Array with `dim_count` elements. Can contain negative values.
     * \param[in] dim_count Number of elements in `dim_values`
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtTensorTypeAndShapeInfo* info, const(long)* dim_values, size_t dim_count) SetDimensions;

    /** \brief Get element type in ::OrtTensorTypeAndShapeInfo
     *
     * \see OrtApi::SetTensorElementType
     *
     * \param[in] info
     * \param[out] out
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtTensorTypeAndShapeInfo)* info,
        ONNXTensorElementDataType* out_) GetTensorElementType;

    /** \brief Get dimension count in ::OrtTensorTypeAndShapeInfo
     *
     * \see OrtApi::GetDimensions
     *
     * \param[in] info
     * \param[out] out
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtTensorTypeAndShapeInfo)* info, size_t* out_) GetDimensionsCount;

    /** \brief Get dimensions in ::OrtTensorTypeAndShapeInfo
     *
     * \param[in] info
     * \param[out] dim_values Array with `dim_values_length` elements. On return, filled with the dimensions stored in the ::OrtTensorTypeAndShapeInfo
     * \param[in] dim_values_length Number of elements in `dim_values`. Use OrtApi::GetDimensionsCount to get this value
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtTensorTypeAndShapeInfo)* info,
        long* dim_values,
        size_t dim_values_length) GetDimensions;

    /** \brief Get symbolic dimension names in ::OrtTensorTypeAndShapeInfo
     *
     * \param[in] info
     * \param[in] dim_params Array with `dim_params_length` elements. On return filled with pointers to null terminated strings of the dimension names
     * \param[in] dim_params_length Number of elements in `dim_params`. Use OrtApi::GetDimensionsCount to get this value
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtTensorTypeAndShapeInfo)* info,
        const(char)*[] dim_params,
        size_t dim_params_length) GetSymbolicDimensions;

    /** \brief Get total number of elements in a tensor shape from an ::OrtTensorTypeAndShapeInfo
     *
     * Return the number of elements specified by the tensor shape (all dimensions multiplied by each other).
     * For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
     *
     * Examples:<br>
     * [] = 1<br>
     * [1,3,4] = 12<br>
     * [2,0,4] = 0<br>
     * [-1,3,4] = -1<br>
     *
     * \param[in] info
     * \param[out] out Number of elements
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtTensorTypeAndShapeInfo)* info, size_t* out_) GetTensorShapeElementCount;

    /// @}
    /// \name OrtValue
    /// @{

    /** \brief Get type and shape information from a tensor ::OrtValue
     *
     * \param[in] value Must be a tensor (not a map/sequence/etc) or will return failure
     * \param[out] out Newly created ::OrtTensorTypeAndShapeInfo. Must be freed with OrtApi::ReleaseTensorTypeAndShapeInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, OrtTensorTypeAndShapeInfo** out_) GetTensorTypeAndShape;

    /** \brief Get type information of an OrtValue
     *
     * \param[in] value
     * \param[out] out Newly created ::OrtTypeInfo. Must be freed with OrtApi::ReleaseTypeInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, OrtTypeInfo** out_) GetTypeInfo;

    /** \brief Get ONNXType of an ::OrtValue
     *
     * \param[in] value
     * \param[out] out
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, ONNXType* out_) GetValueType;

    /// @}
    /// \name OrtMemoryInfo
    /// @{

    /** \brief Create an ::OrtMemoryInfo
     *
     * \param[in] name
     * \param[in] type
     * \param[in] id
     * \param[in] mem_type
     * \param[out] out Newly created ::OrtMemoryInfo. Must be freed with OrtAPi::ReleaseMemoryInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(char)* name,
        OrtAllocatorType type,
        int id,
        OrtMemType mem_type,
        OrtMemoryInfo** out_) CreateMemoryInfo;

    /** \brief Create an ::OrtMemoryInfo for CPU memory
     *
     * Special case version of OrtApi::CreateMemoryInfo for CPU based memory. Same as using OrtApi::CreateMemoryInfo with name = "Cpu" and id = 0.
     *
     * \param[in] type
     * \param[in] mem_type
     * \param[out] out
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtAllocatorType type,
        OrtMemType mem_type,
        OrtMemoryInfo** out_) CreateCpuMemoryInfo;

    /** \brief Compare ::OrtMemoryInfo objects for equality
     *
     * Compares all settings of each ::OrtMemoryInfo for equality
     *
     * \param[in] info1
     * \param[in] info2
     * \param[out] out Set to 0 if equal, -1 if not equal
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtMemoryInfo)* info1, const(OrtMemoryInfo)* info2, int* out_) CompareMemoryInfo;

    /** \brief Get name from ::OrtMemoryInfo
     *
     * \param[in] ptr
     * \param[out] out Writes null terminated string to this pointer. Do NOT free the returned pointer. It is valid for the lifetime of the ::OrtMemoryInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtMemoryInfo)* ptr, const(char*)* out_) MemoryInfoGetName;

    /** \brief Get the id from ::OrtMemoryInfo
     */
    OrtStatusPtr function(const(OrtMemoryInfo)* ptr, int* out_) MemoryInfoGetId;

    /** \brief Get the ::OrtMemType from ::OrtMemoryInfo
     */
    OrtStatusPtr function(const(OrtMemoryInfo)* ptr, OrtMemType* out_) MemoryInfoGetMemType;

    /** \brief Get the ::OrtAllocatorType from ::OrtMemoryInfo
     */
    OrtStatusPtr function(const(OrtMemoryInfo)* ptr, OrtAllocatorType* out_) MemoryInfoGetType;

    /// @}
    /// \name OrtAllocator
    /// @{

    /// \brief Calls OrtAllocator::Alloc function
    OrtStatusPtr function(OrtAllocator* ort_allocator, size_t size, void** out_) AllocatorAlloc;
    /// \brief Calls OrtAllocator::Free function
    OrtStatusPtr function(OrtAllocator* ort_allocator, void* p) AllocatorFree;
    /// \brief Calls OrtAllocator::Info function
    OrtStatusPtr function(const(OrtAllocator)* ort_allocator, const(OrtMemoryInfo*)* out_) AllocatorGetInfo;

    /** \brief Get the default allocator
     *
     * The default allocator is a CPU based, non-arena. Always returns the same pointer to the same default allocator.
     *
     * \param[out] out Returned value should NOT be freed
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtAllocator** out_) GetAllocatorWithDefaultOptions;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Override session symbolic dimensions
     *
     * Override symbolic dimensions (by specific denotation strings) with actual values if known at session initialization time to enable
     * optimizations that can take advantage of fixed values (such as memory planning, etc)
     *
     * \param[in] options
     * \param[in] dim_denotation
     * \param[in] dim_value
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(char)* dim_denotation,
        long dim_value) AddFreeDimensionOverride;

    /// @}
    /// \name OrtValue
    /// @{

    /* Internal information (not seen in Doxygen)
     *
     * APIs to support non-tensor types - map and sequence.
     * Currently only the following types are supported
     * Note: the following types should be kept in sync with data_types.h
     * Map types
     * =========
     * std::map<std::string, std::string>
     * std::map<std::string, int64_t>
     * std::map<std::string, float>
     * std::map<std::string, double>
     * std::map<int64_t, std::string>
     * std::map<int64_t, int64_t>
     * std::map<int64_t, float>
     * std::map<int64_t, double>
     *
     * Sequence types
     * ==============
     * std::vector<std::string>
     * std::vector<int64_t>
     * std::vector<float>
     * std::vector<double>
     * std::vector<std::map<std::string, float>>
     * std::vector<std::map<int64_t, float>
     */

    /** \brief Get non tensor data from an ::OrtValue
     *
     * If `value` is of type ONNX_TYPE_MAP, you need to retrieve the keys and values
     * separately. Use index=0 to retrieve keys and index=1 to retrieve values.
     * If `value` is of type ONNX_TYPE_SEQUENCE, use index to retrieve the index'th element
     * of the sequence.
     *
     * \param[in] value
     * \param[in] index See above for usage based on `value` type
     * \param[in] allocator Allocator used to allocate ::OrtValue
     * \param[out] out Created ::OrtValue that holds the element requested. Must be freed with OrtApi::ReleaseValue
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtValue)* value,
        int index,
        OrtAllocator* allocator,
        OrtValue** out_) GetValue;

    /** \brief Get non tensor value count from an ::OrtValue
     *
     * If `value` is of type ONNX_TYPE_MAP 2 will always be returned. For ONNX_TYPE_SEQUENCE
     * the number of elements in the sequence will be returned
     *
     * \param[in] value
     * \param[out] out
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, size_t* out_) GetValueCount;

    /** \brief Create a map or sequence ::OrtValue
     *
     * To construct a map (ONNX_TYPE_MAP), use num_values = 2 and `in` should be an array of 2 ::OrtValue%s
     * representing keys and values.<br>
     *
     * To construct a sequence (ONNX_TYPE_SEQUENCE), use num_values = N where N is the number of the elements in the
     * sequence. 'in' should be an array of N ::OrtValue%s.
     *
     * \param[in] in See above for details
     * \param[in] num_values
     * \param[in] value_type Must be either ONNX_TYPE_MAP or ONNX_TYPE_SEQUENCE
     * \param[out] out Newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtValue*)* in_,
        size_t num_values,
        ONNXType value_type,
        OrtValue** out_) CreateValue;

    /** \brief Create an opaque (custom user defined type) ::OrtValue
     *
     * Constructs an ::OrtValue that contains a value of non-standard type created for
     * experiments or while awaiting standardization. ::OrtValue in this case would contain
     * an internal representation of the Opaque type. Opaque types are distinguished from
     * each other by two strings 1) domain and 2) type name. The combination of the two
     * must be unique, so the type representation is properly identified internally. The combination
     * must be properly registered from within ORT at both compile/run time or by another API.
     *
     * To construct the ::OrtValue pass domain and type names, also a pointer to a data container
     * the type of which must be known to both ORT and the client program. That data container may or may
     * not match the internal representation of the Opaque type. The sizeof(data_container) is passed for
     * verification purposes.
     *
     * \param[in] domain_name Null terminated string of the domain name
     * \param[in] type_name Null terminated string of the type name
     * \param[in] data_container User pointer Data to populate ::OrtValue
     * \param[in] data_container_size Size in bytes of what `data_container` points to
     * \param[out] out Newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(char)* domain_name,
        const(char)* type_name,
        const(void)* data_container,
        size_t data_container_size,
        OrtValue** out_) CreateOpaqueValue;

    /** \brief Get internal data from an opaque (custom user defined type) ::OrtValue
     *
     * Copies internal data from an opaque value into a user provided buffer
     *
     * \see OrtApi::CreateOpaqueValue
     *
     * \param[in] domain_name Null terminated string of the domain name
     * \param[in] type_name Null terminated string of the type name
     * \param[in] in The opaque ::OrtValue
     * \param[out] data_container Buffer to copy data into
     * \param[out] data_container_size Size in bytes of the buffer pointed to by data_container. Must match the size of the internal buffer.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(char)* domain_name,
        const(char)* type_name,
        const(OrtValue)* in_,
        void* data_container,
        size_t data_container_size) GetOpaqueValue;

    /// @}
    /// \name OrtKernelInfo
    /// Custom operator APIs.
    /// @{

    /** \brief Get a float stored as an attribute in the graph node
     *
     * \param[in] info ::OrtKernelInfo instance
     * \param[in] name Null terminated string of the name of the attribute
     * \param[out] out Pointer to memory where the attribute will be stored
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        const(char)* name,
        float* out_) KernelInfoGetAttribute_float;

    /** \brief Fetch a 64-bit int stored as an attribute in the graph node
     *
     * \param[in] info ::OrtKernelInfo instance
     * \param[in] name Null terminated string of the name of the attribute
     * \param[out] out Pointer to memory where the attribute will be stored
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        const(char)* name,
        long* out_) KernelInfoGetAttribute_int64;

    /** \brief Fetch a string stored as an attribute in the graph node
     *
     * If `out` is nullptr, the value of `size` is set to the true size of the string
     * attribute, and a success status is returned.
     *
     * If the `size` parameter is greater than or equal to the actual string attribute's size,
     * the value of `size` is set to the true size of the string attribute, the provided memory
     * is filled with the attribute's contents, and a success status is returned.
     *
     * If the `size` parameter is less than the actual string attribute's size and `out`
     * is not nullptr, the value of `size` is set to the true size of the string attribute
     * and a failure status is returned.)
     *
     * \param[in] info ::OrtKernelInfo instance
     * \param[in] name Null terminated string of the name of the attribute
     * \param[out] out Pointer to memory where the attribute will be stored
     * \param[in,out] size See above comments for details
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        const(char)* name,
        char* out_,
        size_t* size) KernelInfoGetAttribute_string;

    /// @}
    /// \name OrtKernelContext
    /// Custom operator APIs.
    /// @{

    /** \brief Used for custom operators, get the input count of a kernel
     *
     * \see ::OrtCustomOp
     */
    OrtStatusPtr function(const(OrtKernelContext)* context, size_t* out_) KernelContext_GetInputCount;

    /** \brief Used for custom operators, get the output count of a kernel
     *
     * \see ::OrtCustomOp
     */
    OrtStatusPtr function(const(OrtKernelContext)* context, size_t* out_) KernelContext_GetOutputCount;

    /** \brief Used for custom operators, get an input of a kernel
     *
     * \see ::OrtCustomOp
     */
    OrtStatusPtr function(
        const(OrtKernelContext)* context,
        size_t index,
        const(OrtValue*)* out_) KernelContext_GetInput;

    /** \brief Used for custom operators, get an output of a kernel
     *
     * \see ::OrtCustomOp
     */
    OrtStatusPtr function(
        OrtKernelContext* context,
        size_t index,
        const(long)* dim_values,
        size_t dim_count,
        OrtValue** out_) KernelContext_GetOutput;

    /// @}
    /// \name OrtEnv
    /// @{
    void function(OrtEnv* input) ReleaseEnv;
    /// @}
    /// \name OrtStatus
    /// @{
    void function(OrtStatus* input) ReleaseStatus;
    /// @}
    /// \name OrtMemoryInfo
    /// @{
    void function(OrtMemoryInfo* input) ReleaseMemoryInfo;
    /// @}
    /// \name OrtSession
    /// @{
    void function(OrtSession* input) ReleaseSession; // Don't call ReleaseSession from Dllmain (because session owns a thread pool)
    /// @}
    /// \name OrtValue
    /// @{
    void function(OrtValue* input) ReleaseValue;
    /// @}
    /// \name OrtRunOptions
    /// @{
    void function(OrtRunOptions* input) ReleaseRunOptions;
    /// @}
    /// \name OrtTypeInfo
    /// @{
    void function(OrtTypeInfo* input) ReleaseTypeInfo;
    /// @}
    /// \name OrtTensorTypeAndShapeInfo
    /// @{
    void function(OrtTensorTypeAndShapeInfo* input) ReleaseTensorTypeAndShapeInfo;
    /// @}
    /// \name OrtSessionOptions
    /// @{
    void function(OrtSessionOptions* input) ReleaseSessionOptions;
    /// @}
    /// \name OrtCustomOpDomain
    /// @{
    void function(OrtCustomOpDomain* input) ReleaseCustomOpDomain;

    /// @}
    /// \name OrtTypeInfo
    /// @{

    /** \brief Get denotation from type information
     *
     * Augments ::OrtTypeInfo to return denotations on the type.
     *
     * This is used by WinML to determine if an input/output is intended to be an Image or a Tensor.
     *
     * \param[in] type_info
     * \param[out] denotation Pointer to the null terminated denotation string is written to this pointer. This pointer is valid until the object is destroyed or the name is changed, do not free.
     * \param[out] len Length in bytes of the string returned in `denotation`
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtTypeInfo)* type_info,
        const char** denotation,
        size_t* len) GetDenotationFromTypeInfo;

    /** \brief Get detailed map information from an ::OrtTypeInfo
     *
     * This augments ::OrtTypeInfo to return an ::OrtMapTypeInfo when the type is a map.
     * The OrtMapTypeInfo has additional information about the map's key type and value type.
     *
     * This is used by WinML to support model reflection APIs.
     *
     * \param[out] type_info
     * \param[out] out A pointer to the ::OrtMapTypeInfo. Do not free this value. If type_info
     *             does not contain a map, this value will be set to nullptr.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtTypeInfo)* type_info,
        const(OrtMapTypeInfo*)* out_) CastTypeInfoToMapTypeInfo;

    /** \brief Cast ::OrtTypeInfo to an ::OrtSequenceTypeInfo
     *
     * This api augments ::OrtTypeInfo to return an ::OrtSequenceTypeInfo when the type is a sequence.
     * The ::OrtSequenceTypeInfo has additional information about the sequence's element type.
     *
     * This is used by WinML to support model reflection APIs.
     *
     * \param[in] type_info
     * \param[out] out A pointer to the OrtSequenceTypeInfo. Do not free this value. If type_info
     *             doesn not contain a sequence, this value will be set to nullptr.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtTypeInfo)* type_info,
        const(OrtSequenceTypeInfo*)* out_) CastTypeInfoToSequenceTypeInfo;

    /// @}
    /// \name OrtMapTypeInfo
    /// @{

    /** \brief Get key type from an ::OrtMapTypeInfo
     *
     * Key types are restricted to being scalar types.
     *
     * This is used by WinML to support model reflection APIs.
     *
     * \param[in] map_type_info
     * \param[out] out
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtMapTypeInfo)* map_type_info, ONNXTensorElementDataType* out_) GetMapKeyType;

    /** \brief Get the value type from an ::OrtMapTypeInfo
     *
     * \param[in] map_type_info
     * \param[out] type_info
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtMapTypeInfo)* map_type_info, OrtTypeInfo** type_info) GetMapValueType;

    /// @}
    /// \name OrtSequenceTypeInfo
    /// @{

    /** \brief Get element type from an ::OrtSequenceTypeInfo
     *
     * This is used by WinML to support model reflection APIs.
     *
     * \param[in] sequence_type_info
     * \param[out] type_info
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtSequenceTypeInfo)* sequence_type_info,
        OrtTypeInfo** type_info) GetSequenceElementType;

    /// @}
    /// \name OrtMapTypeInfo
    /// @{
    void function(OrtMapTypeInfo* input) ReleaseMapTypeInfo;
    /// @}
    /// \name OrtSequenceTypeInfo
    /// @{
    void function(OrtSequenceTypeInfo* input) ReleaseSequenceTypeInfo;

    /// @}
    /// \name OrtSession
    /// @{

    /** \brief End profiling and return filename of the profile data
     *
     * Profiling is turned on through OrtApi::EnableProfiling
     *
     * \param[in] session
     * \param[in] allocator
     * \param[out] out Null terminated string of the filename, allocated using `allocator`. Must be freed using `allocator`
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSession* session, OrtAllocator* allocator, char** out_) SessionEndProfiling;

    /** \brief Get ::OrtModelMetadata from an ::OrtSession
     *
     * \param[in] session
     * \param[out] out Newly created ::OrtModelMetadata. Must be freed using OrtApi::ReleaseModelMetadata
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, OrtModelMetadata** out_) SessionGetModelMetadata;

    /// @}
    /// \name OrtModelMetadata
    /// @{

    /** \brief Get `producer name` from an ::OrtModelMetadata
     *
     * \param[in] model_metadata
     * \param[in] allocator
     * \param[out] value Set to a null terminated string allocated using `allocator`. Must be freed using `allocator`
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtModelMetadata)* model_metadata,
        OrtAllocator* allocator,
        char** value) ModelMetadataGetProducerName;

    /** \brief Get `graph name` from an ::OrtModelMetadata
     *
     * \param[in] model_metadata
     * \param[in] allocator
     * \param[out] value Set to a null terminated string allocated using `allocator`. Must be freed using `allocator`
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtModelMetadata)* model_metadata,
        OrtAllocator* allocator,
        char** value) ModelMetadataGetGraphName;

    /** \brief Get `domain` from an ::OrtModelMetadata
     *
     * \param[in] model_metadata
     * \param[in] allocator
     * \param[out] value Set to a null terminated string allocated using `allocator`. Must be freed using `allocator`
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtModelMetadata)* model_metadata,
        OrtAllocator* allocator,
        char** value) ModelMetadataGetDomain;

    /** \brief Get `description` from an ::OrtModelMetadata
     *
     * \param[in] model_metadata
     * \param[in] allocator
     * \param[out] value Set to a null terminated string allocated using `allocator`. Must be freed using `allocator`
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtModelMetadata)* model_metadata,
        OrtAllocator* allocator,
        char** value) ModelMetadataGetDescription;

    /** \brief Return data for a key in the custom metadata map in an ::OrtModelMetadata
     *
     * \param[in] model_metadata
     * \param[in] allocator
     * \param[in] key Null terminated string
     * \param[out] value Set to a null terminated string allocated using `allocator`. Must be freed using `allocator`
     * `value` will be set to nullptr if the given key is not found in the custom metadata map.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtModelMetadata)* model_metadata,
        OrtAllocator* allocator,
        const(char)* key,
        char** value) ModelMetadataLookupCustomMetadataMap;

    /** \brief Get version number from an ::OrtModelMetadata
     *
     * \param[in] model_metadata
     * \param[out] value Set to the version number
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtModelMetadata)* model_metadata, long* value) ModelMetadataGetVersion;

    void function(OrtModelMetadata* input) ReleaseModelMetadata;

    /// @}
    /// \name OrtEnv
    /// @{

    /** \brief Create an OrtEnv
     *
     * Create an environment with global threadpools that will be shared across sessions.
     * Use this in conjunction with OrtApi::DisablePerSessionThreads or else the session will use
     * its own thread pools.
     *
     * \param[in] log_severity_level The log severity level.
     * \param[in] logid The log identifier.
     * \param[in] tp_options
     * \param[out] out Returned newly created OrtEnv. Must be freed with OrtApi::ReleaseEnv
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtLoggingLevel log_severity_level,
        const(char)* logid,
        const(OrtThreadingOptions)* tp_options,
        OrtEnv** out_) CreateEnvWithGlobalThreadPools;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Use global thread pool on a session
     *
     * Disable using per session thread pool and use the shared global threadpool.
     * This should be used in conjunction with OrtApi::CreateEnvWithGlobalThreadPools.
     *
     * \param[in] options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options) DisablePerSessionThreads;

    /// @}
    /// \name OrtThreadingOptions
    /// @{

    /** \brief Create an ::OrtThreadingOptions
     *
     * \param[out] out Newly created ::OrtThreadingOptions. Must be freed with OrtApi::ReleaseThreadingOptions
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtThreadingOptions** out_) CreateThreadingOptions;

    void function(OrtThreadingOptions* input) ReleaseThreadingOptions;

    /// @}
    /// \name OrtModelMetadata
    /// @{

    /**
     *
     * \param[in] model_metadata
     * \param[in] allocator
     * \param[out] keys Array of null terminated strings (array count = num_keys) allocated using `allocator`.
     *  The strings and the pointer array must be freed using `allocator`
     *  `keys` will be set to nullptr if the custom metadata map is empty.
     * \param[out] num_keys Set to the number of elements in the `keys` array
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtModelMetadata)* model_metadata,
        OrtAllocator* allocator,
        char*** keys,
        long* num_keys) ModelMetadataGetCustomMetadataMapKeys;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /**
     *
     * Override symbolic dimensions (by specific name strings) with actual values
     * if known at session initialization time to enable optimizations that can
     * take advantage of fixed values (such as memory planning, etc)
     *
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(char)* dim_name,
        long dim_value) AddFreeDimensionOverrideByName;

    /// @}
    /// \name Misc
    /// @{

    /** \brief Get the names of all available providers
     *
     * \note The providers in the list are not guaranteed to be usable. They may fail to load due to missing system dependencies.
     *    For example, if the CUDA/cuDNN libraries are not installed, the CUDA provider will report an error when it is added to the session options.
     *
     * \param[out] out_ptr Set to a pointer to an array of null terminated strings of the available providers. The entries and the
     *    array itself must be freed using OrtApi::ReleaseAvailableProviders
     * \param[out] provider_length Set to the number of entries in the `out_ptr` array
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(char*** out_ptr, int* provider_length) GetAvailableProviders;

    /** \brief Release data from OrtApi::GetAvailableProviders. This API will never fail
     * so you can rely on it in a noexcept code.
     *
     * \param[in] ptr The `out_ptr` result from OrtApi::GetAvailableProviders.
     * \param[in] providers_length The `provider_length` result from OrtApi::GetAvailableProviders
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        char** ptr,
        int providers_length) ReleaseAvailableProviders;

    /// @}
    /// \name OrtValue
    /// @{

    /** \brief Get the length of a single string in a string tensor
     *
     * \param[in] value A string tensor
     * \param[in] index Index of the string in the tensor
     * \param[out] out Set to number of bytes of the string element
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, size_t index, size_t* out_) GetStringTensorElementLength;

    /** \brief Get a single string from a string tensor
     *
     * \param[in] value A string tensor
     * \param[in] s_len Number of bytes in the `s` buffer. Must match the value returned by OrtApi::GetStringTensorElementLength.
     * \param[in] index Index of the string in the tensor
     * \param[out] s The string element contents in UTF-8 encoding. The string is NOT null-terminated.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, size_t s_len, size_t index, void* s) GetStringTensorElement;

    /** \brief Set a single string in a string tensor
     *
     * \param[in] value A string tensor
     * \param[in] s A null terminated UTF-8 encoded string
     * \param[in] index Index of the string in the tensor to set
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtValue* value, const(char)* s, size_t index) FillStringTensorElement;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Set a session configuration entry as a pair of strings
     *
     * If a configuration with same key exists, this will overwrite the configuration with the given config_value.
     *
     * The config_key and the format of config_value are defined in onnxruntime_session_options_config_keys.h
     *
     * \param[in] options
     * \param[in] config_key A null terminated string representation of the config key
     * \param[in] config_value A null terminated string representation of the config value
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(char)* config_key,
        const(char)* config_value) AddSessionConfigEntry;

    /// @}
    /// \name OrtAllocator
    /// @{

    /** \brief Create an allocator for an ::OrtSession following an ::OrtMemoryInfo
     *
     * \param[in] session
     * \param[in] mem_info valid ::OrtMemoryInfo instance
     * \param[out] out Newly created ::OrtAllocator. Must be freed with OrtApi::ReleaseAllocator
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtSession)* session,
        const(OrtMemoryInfo)* mem_info,
        OrtAllocator** out_) CreateAllocator;

    /** \brief Release an ::OrtAllocator obtained from OrtApi::CreateAllocator
     */
    void function(OrtAllocator* input) ReleaseAllocator;

    /// @}
    /// \name OrtSession
    /// @{

    /** \brief Run a model using Io Bindings for the inputs & outputs
     *
     * \see OrtApi::Run
     *
     * \param[in] session
     * \param[in] run_options
     * \param[in] binding_ptr
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSession* session, const(OrtRunOptions)* run_options, const(
            OrtIoBinding)* binding_ptr) RunWithBinding;

    /** \brief Create an ::OrtIoBinding instance
     *
     * An IoBinding object allows one to bind pre-allocated ::OrtValue%s to input names.
     * Thus if you want to use a raw on device buffer as input or output you can avoid
     * extra copy during runtime.
     *
     * \param[in] session
     * \param[out] out Newly created ::OrtIoBinding. Must be freed with OrtApi::ReleaseIoBinding
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSession* session, OrtIoBinding** out_) CreateIoBinding;

    /// @}
    /// \name OrtIoBinding
    /// @{

    /** \brief Release an ::OrtIoBinding obtained from OrtApi::CreateIoBinding
     */
    void function(OrtIoBinding* input) ReleaseIoBinding;

    /** \brief Bind an ::OrtValue to an ::OrtIoBinding input
     *
     * When using OrtApi::RunWithBinding this value is used for the named input
     *
     * \param[in] binding_ptr
     * \param[in] name Name for the model input
     * \param[in] val_ptr ::OrtValue of Tensor type.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtIoBinding* binding_ptr, const(char)* name, const(OrtValue)* val_ptr) BindInput;

    /** \brief Bind an ::OrtValue to an ::OrtIoBinding output
     *
     * When using OrtApi::RunWithBinding this value is used for the named output
     *
     * \param[in] binding_ptr
     * \param[in] name Null terminated string of the model output name
     * \param[in] val_ptr ::OrtValue of Tensor type.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtIoBinding* binding_ptr, const(char)* name, const(OrtValue)* val_ptr) BindOutput;

    /** \brief Bind an ::OrtIoBinding output to a device
     *
     * Binds the ::OrtValue to a device which is specified by ::OrtMemoryInfo.
     * You can either create an instance of ::OrtMemoryInfo with a device id or obtain one from the allocator that you have created/are using
     * This is useful when one or more outputs have dynamic shapes and, it is hard to pre-allocate and bind a chunk of
     * memory within ::OrtValue ahead of time.
     *
     * \see OrtApi::RunWithBinding
     *
     * \param[in] binding_ptr
     * \param[in] name Null terminated string of the device name
     * \param[in] mem_info_ptr
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtIoBinding* binding_ptr, const(char)* name, const(OrtMemoryInfo)* mem_info_ptr) BindOutputToDevice;

    /** \brief Get the names of an ::OrtIoBinding's outputs
     *
     * Returns the names of the outputs in the order they were bound. This is useful after running the model
     * with bound outputs because the returned names are in order in which output ::OrtValue are returned. This is useful if
     * the order of outputs and their names is not known.
     *
     * \param[in] binding_ptr
     * \param[in] allocator Allocator used to allocate continuous buffers for output strings and lengths.
     * \param[out] buffer Returns an array of non-null terminated UTF-8 strings. The number of strings stored is returned in the count parameter.
     *   This buffer is allocated using `allocator` and must be freed using it.
     * \param[out] lengths Returns an array of `count` lengths of the strings returned in `buffer`
     *   This buffer is allocated using `allocator` and must be freed using it.
     * \param[out] count Number of strings returned. If `binding_ptr` has no bound outputs, zero is returned,
     *              no memory allocation is performed and buffer and lengths are set to nullptr.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtIoBinding)* binding_ptr,
        OrtAllocator* allocator,
        char** buffer,
        size_t** lengths,
        size_t* count) GetBoundOutputNames;

    /** \brief Get the output ::OrtValue objects from an ::OrtIoBinding
     *
     * Returns an array of pointers to individually allocated ::OrtValue%s that contain results of a model execution with OrtApi::RunWithBinding
     * The array contains the same number of ::OrtValue%s and they are in the same order as they were bound with OrtApi::BindOutput
     * or OrtApi::BindOutputToDevice.
     *
     * The returned ::OrtValue%s must be released using OrtApi::ReleaseValue after they are no longer needed.
     * The array is allocated using the specified instance of the allocator and must be freed using the same allocator after
     * all the ::OrtValue%s contained therein are individually released.
     *
     * \param[in] binding_ptr
     * \param[in] allocator Allocator used to allocate output array
     * \param[out] output Set to the allocated array of allocated ::OrtValue outputs. Set to nullptr if there are 0 outputs.
     * \param[out] output_count Set to number of ::OrtValue%s returned
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtIoBinding)* binding_ptr,
        OrtAllocator* allocator,
        OrtValue*** output,
        size_t* output_count) GetBoundOutputValues;

    /** \brief Clears any previously set Inputs for an ::OrtIoBinding
     */
    void function(OrtIoBinding* binding_ptr) ClearBoundInputs;

    /** \brief Clears any previously set Outputs for an ::OrtIoBinding
     */
    void function(OrtIoBinding* binding_ptr) ClearBoundOutputs;

    /// @}
    /// \name OrtValue
    /// @{

    /** \brief Direct memory access to a specified tensor element
     *
     * For example, given a tensor with shape of [3,224,224], a pointer to the element at location [2,150,128] can be retrieved
     *
     * This function only works for numeric type tensors (No strings, etc).
     * This is a no-copy method whose returned pointer is valid until the passed in ::OrtValue is free'd.
     *
     * \param[in] value
     * \param[in] location_values Pointer to an array of index values that specify an element's location relative to its shape
     * \param[in] location_values_count Number of elements in location_values. Must match the number of elements in the tensor's shape.
     * \param[out] out Set to a pointer to the element specified
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtValue* value, const(long)* location_values, size_t location_values_count, void** out_) TensorAt;

    /// @}
    /// \name OrtEnv
    /// @{

    /** \brief Create an allocator and register it with the ::OrtEnv
     *
     * Enables sharing the allocator between multiple sessions that use the same env instance.
     * Lifetime of the created allocator will be valid for the duration of the environment.
     * Returns an error if an allocator with the same ::OrtMemoryInfo is already registered.
     *
     * See https://onnxruntime.ai/docs/get-started/with-c.html for details.
     *
     * \param[in] env ::OrtEnv instance
     * \param[in] mem_info
     * \param[in] arena_cfg Pass nullptr for defaults
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtEnv* env,
        const(OrtMemoryInfo)* mem_info,
        const(OrtArenaCfg)* arena_cfg) CreateAndRegisterAllocator;

    /** \brief Set language projection
     *
     * Set the language projection for collecting telemetry data when Env is created.
     *
     * The default is ORT_PROJECTION_C, which means it will classify the language not in the list to C also.
     *
     * \param[in] ort_env
     * \param[in] projection
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtEnv)* ort_env, OrtLanguageProjection projection) SetLanguageProjection;

    /// @}
    /// \name OrtSession
    /// @{

    /** \brief Return the time that profiling was started
     *
     * \note The timer precision varies per platform. On Windows and MacOS, the precision will be ~100ns
     *
     * \param[in] session
     * \param[out] out nanoseconds of profiling's start time
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtSession)* session, ulong* out_) SessionGetProfilingStartTimeNs;

    /// @}
    /// \name OrtThreadingOptions
    /// @{

    /** \brief Set global intra-op thread count
     *
     * This configures the global thread pool options to be used in the call to OrtApi::CreateEnvWithGlobalThreadPools
     *
     * \param[in] tp_options
     * \param[in] intra_op_num_threads Number of threads, special values:<br>
     *    0 = Use default thread count<br>
     *    1 = The invoking thread will be used; no threads will be created in the thread pool.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtThreadingOptions* tp_options, int intra_op_num_threads) SetGlobalIntraOpNumThreads;

    /** \brief Set global inter-op thread count
     *
     * This configures the global thread pool options to be used in the call to OrtApi::CreateEnvWithGlobalThreadPools
     *
     * \param[in] tp_options
     * \param[in] inter_op_num_threads Number of threads, special values:<br>
     *    0 = Use default thread count<br>
     *    1 = The invoking thread will be used; no threads will be created in the thread pool.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtThreadingOptions* tp_options, int inter_op_num_threads) SetGlobalInterOpNumThreads;

    /** \brief Set global spin control options
     *
     * This will configure the global thread pool options to be used in the call to OrtApi::CreateEnvWithGlobalThreadPools.
     * Allow spinning of thread pools when their queues are empty. This will set the value for both
     * inter_op and intra_op threadpools.
     *
     * \param[in] tp_options
     * \param[in] allow_spinning Valid values are 0 or 1.<br>
     *   0 = It won't spin (recommended if CPU usage is high)<br>
     *   1 = Threadpool will spin to wait for queue to become non-empty
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtThreadingOptions* tp_options, int allow_spinning) SetGlobalSpinControl;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Add a pre-allocated initializer to a session
     *
     * If a model contains an initializer with a name that is same as the name passed to this call,
     * ORT will use this initializer instance instead of deserializing one from the model file. This
     * is useful when you want to share the same initializer across sessions.
     *
     * \param[in] options
     * \param[in] name Null terminated string of the initializer name
     * \param[in] val ::OrtValue containing the initializer. Its lifetime and the underlying initializer buffer must be
     *   managed by the user (created using the OrtApi::CreateTensorWithDataAsOrtValue) and it must outlive the session object
     *   to which it is added.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(char)* name,
        const(OrtValue)* val) AddInitializer;

    /// @}
    /// \name OrtEnv
    /// @{

    /**
     * Create a custom environment with global threadpools and logger that will be shared across sessions.
     * Use this in conjunction with OrtApi::DisablePerSessionThreads or else the session will use
     * its own thread pools.
     *
     * \param[in] logging_function A pointer to a logging function.
     * \param[in] logger_param A pointer to arbitrary data passed as the ::OrtLoggingFunction `param` parameter to
     *                         `logging_function`.
     * \param[in] log_severity_level The log severity level.
     * \param[in] logid The log identifier.
     * \param[in] tp_options
     * \param[out] out Newly created OrtEnv. Must be freed with OrtApi::ReleaseEnv
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtLoggingFunction logging_function,
        void* logger_param,
        OrtLoggingLevel log_severity_level,
        const(char)* logid,
        const(OrtThreadingOptions)* tp_options,
        OrtEnv** out_) CreateEnvWithCustomLoggerAndGlobalThreadPools;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Append CUDA provider to session options
     *
     * If CUDA is not available (due to a non CUDA enabled build, or if CUDA is not installed on the system), this function will return failure.
     *
     * \param[in] options
     * \param[in] cuda_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(OrtCUDAProviderOptions)* cuda_options) SessionOptionsAppendExecutionProvider_CUDA;

    /** \brief Append ROCM execution provider to the session options
     *
     * If ROCM is not available (due to a non ROCM enabled build, or if ROCM is not installed on the system), this function will return failure.
     *
     * \param[in] options
     * \param[in] rocm_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(OrtROCMProviderOptions)* rocm_options) SessionOptionsAppendExecutionProvider_ROCM;

    /** \brief Append OpenVINO execution provider to the session options
     *
     * If OpenVINO is not available (due to a non OpenVINO enabled build, or if OpenVINO is not installed on the system), this function will fail.
     *
     * \param[in] options
     * \param[in] provider_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(OrtOpenVINOProviderOptions)* provider_options) SessionOptionsAppendExecutionProvider_OpenVINO;

    /// @}
    /// \name OrtThreadingOptions
    /// @{

    /** \brief Set threading flush-to-zero and denormal-as-zero
     *
     * Sets global thread pool options to be used in the call to OrtApi::CreateEnvWithGlobalThreadPools.
     * Flush-to-zero and denormal-as-zero are applied to threads in both intra and inter global thread pool.
     * \note This option is not needed if the models used have no denormals. Having no denormals is recommended as this option may hurt model accuracy.
     *
     * \param[in] tp_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtThreadingOptions* tp_options) SetGlobalDenormalAsZero;

    /// @}
    /// \name OrtArenaCfg
    /// @{

    /** \deprecated Use OrtApi::CreateArenaCfgV2
     *
     * This will create the configuration of an arena that can eventually be used to define an arena based allocator's behavior
     *
     * \param[in] max_mem Use 0 to allow ORT to choose the default
     * \param[in] arena_extend_strategy Use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
     * \param[in] initial_chunk_size_bytes Use -1 to allow ORT to choose the default
     * \param[in] max_dead_bytes_per_chunk Use -1 to allow ORT to choose the default
     * \param[in] out A pointer to an OrtArenaCfg instance
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        size_t max_mem,
        int arena_extend_strategy,
        int initial_chunk_size_bytes,
        int max_dead_bytes_per_chunk,
        OrtArenaCfg** out_) CreateArenaCfg;

    void function(OrtArenaCfg* input) ReleaseArenaCfg;

    /// @}
    /// \name OrtModelMetadata
    /// @{

    /**
     * Use this to obtain the description of the graph present in the model
     * (doc_string field of the GraphProto message within the ModelProto message).
     * If it doesn't exist, an empty string will be returned.
     *
     * \param[in] model_metadata An instance of ::OrtModelMetadata
     * \param[in] allocator Allocator used to allocate the string that will be returned back
     * \param[out] value Set to a null terminated string allocated using `allocator`.  The caller is responsible for freeing it using `allocator`
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtModelMetadata)* model_metadata,
        OrtAllocator* allocator,
        char** value) ModelMetadataGetGraphDescription;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Append TensorRT provider to session options
     *
     * If TensorRT is not available (due to a non TensorRT enabled build, or if TensorRT is not installed on the system), this function will return failure.
     *
     * \param[in] options
     * \param[in] tensorrt_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(OrtTensorRTProviderOptions)* tensorrt_options) SessionOptionsAppendExecutionProvider_TensorRT;

    /// @}
    /// \name Misc
    /// @{

    /** \brief Set current GPU device ID
     *
     * Set the current device id of the GPU execution provider (CUDA/tensorrt/rocm). The device id should be less
     * than the total number of devices available. This is only useful when multiple-GPUs are installed and it is
     * required to restrict execution to a single GPU.
     *
     * \param[in] device_id
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(int device_id) SetCurrentGpuDeviceId;

    /** \brief Get current GPU device ID
     *
     * Get the current device id of the GPU execution provider (CUDA/tensorrt/rocm).
     *
     * \see OrtApi::SetCurrentGpuDeviceId
     *
     * \param[out] device_id
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(int* device_id) GetCurrentGpuDeviceId;

    /// @}
    /// \name OrtKernelInfo
    /// Custom operator APIs.
    /// @{

    /** \brief Fetch an array of int64_t values stored as an attribute in the graph node
     *
     *
     * If `out` is nullptr, the value of `size` is set to the true size of the attribute
     * array's size, and a success status is returned.
     *
     * If the `size` parameter is greater than or equal to the actual attribute array's size,
     * the value of `size` is set to the true size of the attribute array's size,
     * the provided memory is filled with the attribute's contents,
     * and a success status is returned.
     *
     * If the `size` parameter is less than the actual attribute array's size and `out`
     * is not nullptr, the value of `size` is set to the true size of the attribute array's size
     * and a failure status is returned.)
     *
     * \param[in] info instance
     * \param[in] name name of the attribute to be parsed
     * \param[out] out pointer to memory where the attribute's contents are to be stored
     * \param[in, out] size actual size of attribute array
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        const(char)* name,
        float* out_,
        size_t* size) KernelInfoGetAttributeArray_float;

    /** \brief Fetch an array of int64_t values stored as an attribute in the graph node
     *
     * If `out` is nullptr, the value of `size` is set to the true size of the attribute
     * array's size, and a success status is returned.
     *
     * If the `size` parameter is greater than or equal to the actual attribute array's size,
     * the value of `size` is set to the true size of the attribute array's size,
     * the provided memory is filled with the attribute's contents,
     * and a success status is returned.
     *
     * If the `size` parameter is less than the actual attribute array's size and `out`
     * is not nullptr, the value of `size` is set to the true size of the attribute array's size
     * and a failure status is returned.)
     *
     * \param[in] info instance
     * \param[in] name name of the attribute to be parsed
     * \param[out] out pointer to memory where the attribute's contents are to be stored
     * \param[in, out] size actual size of attribute array
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        const(char)* name,
        long* out_,
        size_t* size) KernelInfoGetAttributeArray_int64;

    /// @}
    /// \name OrtArenaCfg
    /// @{

    /** \brief Create an ::OrtArenaCfg
     *
     * Create the configuration of an arena that can eventually be used to define an arena based allocator's behavior.
     *
     * Supported keys are (See https://onnxruntime.ai/docs/get-started/with-c.html for details on what the
     * following parameters mean and how to choose these values.):
     * "max_mem": Maximum memory that can be allocated by the arena based allocator.
     *  Use 0 for ORT to pick the best value. Default is 0.
     * "arena_extend_strategy": 0 = kNextPowerOfTwo, 1 = kSameAsRequested.
     *  Use -1 to allow ORT to choose the default.
     * "initial_chunk_size_bytes": (Possible) Size of the first allocation in the arena.
     *  Only relevant if arena strategy is `kNextPowerOfTwo`. Use -1 to allow ORT to choose the default.
     *  Ultimately, the first allocation size is determined by the allocation memory request.
     * "max_dead_bytes_per_chunk": Threshold of unused memory in an allocated chunk of arena memory after
     *  crossing which the current chunk is chunked into 2.
     * "initial_growth_chunk_size_bytes": (Possible) Size of the second allocation in the arena.
     *  Only relevant if arena strategy is `kNextPowerOfTwo`. Use -1 to allow ORT to choose the default.
     * "max_power_of_two_extend_bytes": The maximum enxtend size if arena strategy is `kNextPowerOfTwo`.
     *  It is not an allocation limit, it is only a limit for extention when requested byte is less than the limit.
     *  When requested bytes is more than the limit, allocator will still return as requested.
     *  Use -1 to allow ORT to choose the default 1GB for max_power_of_two_extend_bytes.
     *  Ultimately, the allocation size is determined by the allocation memory request.
     *  Further allocation sizes are governed by the arena extend strategy.
     *
     * \param[in] arena_config_keys Keys to configure the arena
     * \param[in] arena_config_values Values to configure the arena
     * \param[in] num_keys Number of keys in `arena_config_keys` and `arena_config_values`
     * \param[out] out Newly created ::OrtArenaCfg. Must be freed with OrtApi::ReleaseArenaCfg
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(char*)* arena_config_keys,
        const(size_t)* arena_config_values,
        size_t num_keys,
        OrtArenaCfg** out_) CreateArenaCfgV2;

    /// @}
    /// \name OrtRunOptions
    /// @{

    /** \brief Set a single run configuration entry as a pair of strings
     *
     * If a configuration with same key exists, this will overwrite the configuration with the given config_value
     *
     * The config_key and the format of config_value are defined in onnxruntime_run_options_config_keys.h
     *
     * \param[in] options
     * \param[in] config_key A null terminated string representation of the config key
     * \param[in] config_value  A null terminated string representation of the config value
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtRunOptions* options,
        const(char)* config_key,
        const(char)* config_value) AddRunConfigEntry;

    /// @}
    /// \name OrtPrepackedWeightsContainer
    /// @{

    /** \brief Create an ::OrtPrepackedWeightsContainer
     *
     * This container will hold pre-packed buffers of shared initializers for sharing between sessions
     * (i.e.) if there are shared initializers that can be shared between sessions, the pre-packed buffers
     * of these (if any) may possibly be shared to provide memory footprint savings. Pass this container
     * to sessions that you would like to share pre-packed buffers of shared initializers at session
     * creation time.
     *
     *  \param[out] out Newly created ::OrtPrepackedWeightsContainer. Must be freed with OrtApi::ReleasePrepackedWeightsContainer
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtPrepackedWeightsContainer** out_) CreatePrepackedWeightsContainer;

    /** \brief Release OrtPrepackedWeightsContainer instance
     *
     * \note instance must not be released until the sessions using it are released
     */
    void function(OrtPrepackedWeightsContainer* input) ReleasePrepackedWeightsContainer;

    /// @}
    /// \name OrtSession
    /// @{

    /** \brief Create session with prepacked weights container
     *
     * Same functionality offered by OrtApi::CreateSession except that a container that contains
     * pre-packed weights' buffers is written into/read from by the created session.
     * This is useful when used in conjunction with OrtApi::AddInitializer which injects
     * shared initializer info into sessions. Wherever possible, the pre-packed versions of these
     * shared initializers are cached in this container so that multiple sessions can just re-use
     * these instead of duplicating these in memory.
     *
     * \param[in] env OrtEnv instance instance
     * \param[in] model_path Null terminated string of the path (wchar on Windows, char otherwise)
     * \param[in] options
     * \param[in] prepacked_weights_container
     * \param[out] out Newly created ::OrtSession. Must be freed with OrtApi::ReleaseSession
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtEnv)* env,
        const(char)* model_path,
        const(OrtSessionOptions)* options,
        OrtPrepackedWeightsContainer* prepacked_weights_container,
        OrtSession** out_) CreateSessionWithPrepackedWeightsContainer;

    /** \brief Create session from memory with prepacked weights container
     *
     * Same functionality offered by OrtApi::CreateSessionFromArray except that a container that contains
     * pre-packed weights' buffers is written into/read from by the created session.
     * This is useful when used in conjunction with OrtApi::AddInitializer which injects
     * shared initializer info into sessions. Wherever possible, the pre-packed versions of these
     * shared initializers are cached in this container so that multiple sessions can just re-use
     * these instead of duplicating these in memory.
     *
     * \param[in] env
     * \param[in] model_data Array of bytes holding the model
     * \param[in] model_data_length Number of bytes in `model_data_model`
     * \param[in] options
     * \param[in] prepacked_weights_container
     * \param[out] out Newly created ::OrtSession. Must be freed with OrtApi::ReleaseSession
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtEnv)* env,
        const(void)* model_data,
        size_t model_data_length,
        const(OrtSessionOptions)* options,
        OrtPrepackedWeightsContainer* prepacked_weights_container,
        OrtSession** out_) CreateSessionFromArrayWithPrepackedWeightsContainer;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Append TensorRT execution provider to the session options
     *
     * If TensorRT is not available (due to a non TensorRT enabled build), this function will return failure.
     *
     * This is slightly different from OrtApi::SessionOptionsAppendExecutionProvider_TensorRT, it takes an
     * ::OrtTensorRTProviderOptions which is publicly defined. This takes an opaque ::OrtTensorRTProviderOptionsV2
     * which must be created with OrtApi::CreateTensorRTProviderOptions.
     *
     * For OrtApi::SessionOptionsAppendExecutionProvider_TensorRT, the user needs to instantiate ::OrtTensorRTProviderOptions
     * as well as allocate/release buffers for some members of ::OrtTensorRTProviderOptions.
     * Here, OrtApi::CreateTensorRTProviderOptions and Ortapi::ReleaseTensorRTProviderOptions will do the memory management for you.
     *
     * \param[in] options
     * \param[in] tensorrt_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(OrtTensorRTProviderOptionsV2)* tensorrt_options) SessionOptionsAppendExecutionProvider_TensorRT_V2;

    /// @}
    /// \name OrtTensorRTProviderOptionsV2
    /// @{

    /** \brief Create an OrtTensorRTProviderOptionsV2
     *
     * \param[out] out Newly created ::OrtTensorRTProviderOptionsV2. Must be released with OrtApi::ReleaseTensorRTProviderOptions
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtTensorRTProviderOptionsV2** out_) CreateTensorRTProviderOptions;

    /** \brief Set options in a TensorRT Execution Provider.
     *
     * Please refer to https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#cc
     * to know the available keys and values. Key should be in null terminated string format of the member of ::OrtTensorRTProviderOptionsV2
     * and value should be its related range.
     *
     * For example, key="trt_max_workspace_size" and value="2147483648"
     *
     * \param[in] tensorrt_options
     * \param[in] provider_options_keys Array of UTF-8 null-terminated string for provider options keys
     * \param[in] provider_options_values Array of UTF-8 null-terminated string for provider options values
     * \param[in] num_keys Number of elements in the `provider_option_keys` and `provider_options_values` arrays
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtTensorRTProviderOptionsV2* tensorrt_options,
        const(char*)* provider_options_keys,
        const(char*)* provider_options_values,
        size_t num_keys) UpdateTensorRTProviderOptions;

    /** \brief Get serialized TensorRT provider options string.
     *
     * For example, "trt_max_workspace_size=2147483648;trt_max_partition_iterations=10;trt_int8_enable=1;......"
     *
     * \param tensorrt_options - OrtTensorRTProviderOptionsV2 instance
     * \param allocator - a ptr to an instance of OrtAllocator obtained with OrtApi::CreateAllocator or OrtApi::GetAllocatorWithDefaultOptions
     *                      the specified allocator will be used to allocate continuous buffers for output strings and lengths.
     * \param ptr - is a UTF-8 null terminated string allocated using 'allocator'. The caller is responsible for using the same allocator to free it.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtTensorRTProviderOptionsV2)* tensorrt_options, OrtAllocator* allocator, char** ptr) GetTensorRTProviderOptionsAsString;

    /** \brief Release an ::OrtTensorRTProviderOptionsV2
     *
     * \note This is an exception in the naming convention of other Release* functions, as the name of the method does not have the V2 suffix, but the type does
     */
    void function(OrtTensorRTProviderOptionsV2* input) ReleaseTensorRTProviderOptions;

    /// @}
    /// \name OrtSessionOptions
    /// @{

    /** \brief Enable custom operators
     *
     * See onnxruntime-extensions: https://github.com/microsoft/onnxruntime-extensions.git
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options) EnableOrtCustomOps;

    /// @}
    /// \name OrtAllocator
    /// @{

    /** \brief Register a custom allocator
     *
     * Enables sharing between multiple sessions that use the same env instance.
     * Returns an error if an allocator with the same ::OrtMemoryInfo is already registered.
     *
     * The behavior of this is exactly the same as OrtApi::CreateAndRegisterAllocator except
     * instead of ORT creating an allocator based on provided info, in this case
     * ORT uses the user-provided custom allocator.
     * See https://onnxruntime.ai/docs/get-started/with-c.html for details.
     *
     * \param[in] env
     * \param[in] allocator User provided allocator
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtEnv* env, OrtAllocator* allocator) RegisterAllocator;

    /** \brief Unregister a custom allocator
     *
     * It is an error if you provide an ::OrtMemoryInfo not corresponding to any
     * registered allocators for sharing.
     *
     * \param[in] env
     * \param[in] mem_info
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtEnv* env,
        const(OrtMemoryInfo)* mem_info) UnregisterAllocator;

    /// @}
    /// \name OrtValue
    /// @{

    /** \brief Sets *out to 1 iff an ::OrtValue is a SparseTensor, and 0 otherwise
     *
     * \param[in] value existing ::OrtValue
     * \param[out] out unless an error occurs, contains 1 iff the value contains an instance
     *  of sparse tensor or 0 otherwise.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, int* out_) IsSparseTensor;

    /** \brief Create an ::OrtValue with a sparse tensor that is empty.
     *
     * Use FillSparseTensor<Format>() functions to populate sparse tensor with non-zero values and
     * format specific indices data.
     * Use ReleaseValue to destroy the sparse tensor, this will also release the buffer inside the output value
     * if any was allocated.
     * \param[in,out] allocator allocator to use when performing an allocation. Allocation will be performed
     *   by FillSparseTensor<Format>() APIs. The lifespan of the allocator instance must eclipse the lifespan
     *   this sparse tensor instance as the same allocator will be used to free memory.
     * \param[in] dense_shape shape of the original dense tensor
     * \param[in] dense_shape_len number of shape dimensions being passed
     * \param[in] type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
     * \param[out] out Should be freed by calling ReleaseValue
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtAllocator* allocator,
        const(long)* dense_shape,
        size_t dense_shape_len,
        ONNXTensorElementDataType type,
        OrtValue** out_) CreateSparseTensorAsOrtValue;

    /**
     * This fills populates an empty tensor that was created using OrtApi::CreateSparseTensorAsOrtValue.
     * This will allocate required memory and copy the supplied NNZ values and COO indices into that memory allocation.
     * Memory allocation is performed using the allocator that was specified with OrtApi::CreateSparseTensorAsOrtValue.
     *
     * \param[in,out] ort_value ::OrtValue to populate with data
     * \param[in] data_mem_info serves to identify the location of the data to be copied. If the allocator specified
     *  at the creation time has memory info that is not the same as mem_info argument to this function a X-device copy will be performed.
     *  String data is assumed to be on CPU and will only be copied into a CPU allocated buffer.
     * \param[in] values_shape pointer to values shape array
     * \param[in] values_shape_len length of the values_shape
     * \param[in] values pointer to an array of values. For strings, pass const char**.
     * \param[in] indices_data pointer to a location of COO indices
     * \param[in] indices_num number of COO indices
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtValue* ort_value,
        const(OrtMemoryInfo)* data_mem_info,
        const(long)* values_shape,
        size_t values_shape_len,
        const(void)* values,
        const(long)* indices_data,
        size_t indices_num) FillSparseTensorCoo;

    /**
     * This fills populates an empty tensor that was created using OrtApi::CreateSparseTensorAsOrtValue.
     * This will allocate required memory and copy the supplied NNZ values and CSR indices into that memory allocation.
     * Memory allocation is performed using the allocator that was specified with OrtApi::CreateSparseTensorAsOrtValue.
     *
     * \param[in,out] ort_value ::OrtValue to populate with data
     * \param[in] data_mem_info serves to identify the location of the data to be copied. If the allocator specified
     *  at the creation time has memory info that is not the same as mem_info argument to this function a X-device copy will be performed.
     *  String data is assumed to be on CPU and will only be copied into a CPU allocated buffer.
     * \param[in] values_shape pointer to values shape array
     * \param[in] values_shape_len length of the values_shape
     * \param[in] values - pointer to an array of values. For strings, pass const char**.
     * \param[in] inner_indices_data pointer to a location of CSR inner indices
     * \param[in] inner_indices_num number of CSR inner indices
     * \param[in] outer_indices_data pointer to a location of CSR outer indices
     * \param[in] outer_indices_num number of CSR outer indices
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtValue* ort_value,
        const(OrtMemoryInfo)* data_mem_info,
        const(long)* values_shape,
        size_t values_shape_len,
        const(void)* values,
        const(long)* inner_indices_data,
        size_t inner_indices_num,
        const(long)* outer_indices_data,
        size_t outer_indices_num) FillSparseTensorCsr;

    /**
     * This fills populates an empty tensor that was created using OrtApi::CreateSparseTensorAsOrtValue.
     * This will allocate required memory and copy the supplied NNZ values and BlockSparse indices into that memory allocation.
     * Memory allocation is performed using the allocator that was specified with OrtApi::CreateSparseTensorAsOrtValue.
     *
     * \param[in,out] ort_value ::OrtValue to populate with data
     * \param[in] data_mem_info serves to identify the location of the data to be copied. If the allocator specified
     *  at the creation time has memory info that is not the same as mem_info argument to this function a X-device copy will be performed.
     *  String data is assumed to be on CPU and will only be copied into a CPU allocated buffer.
     * \param[in] values_shape
     * \param[in] values_shape_len
     * \param[in] values structure with values information
     * \param[in] indices_shape_data pointer to a location of indices shape
     * \param[in] indices_shape_len length of the block sparse indices shape
     * \param[in] indices_data pointer to a location of indices data. Shape will determine the length of the indices data.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtValue* ort_value,
        const(OrtMemoryInfo)* data_mem_info,
        const(long)* values_shape,
        size_t values_shape_len,
        const(void)* values,
        const(long)* indices_shape_data,
        size_t indices_shape_len,
        const(int)* indices_data) FillSparseTensorBlockSparse;

    /**
     * Create an ::OrtValue with a sparse tensor. This is the first step.
     * Next, use Use<Format>Indices() functions to supply sparse tensor with
     * format specific indices data and set its sparse format to a specific enum value.
     * This will not perform memory allocations. It will
     * use supplied user buffer which should outlive the created sparse tensor.
     * Use OrtApi::ReleaseValue to destroy the sparse tensor. It would not release the supplied values buffer.
     * This function can not be used to map strings from the user allocated memory. Strings must always be copied
     * and have UTF-8 encoding. Therefore, use OrtApi::CreateSparseTensorAsOrtValue above and then fill it with data
     * using appropriate Make*() function.
     *
     * \param[in] info memory info where sparse values reside.
     * \param[in,out] p_data pointer to a user allocated buffer with values. To create a full sparse tensor with no non-zero
     *   values, pass nullptr
     * \param[in] dense_shape shape of the original dense tensor
     * \param[in] dense_shape_len number of shape dimensions being passed
     * \param[in] values_shape shape of the values data. To create a fully sparse tensor with no non-zero values,
     *   pass {0} shape.
     * \param[in] values_shape_len number of values shape dimensions
     * \param[in] type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
     * \param[out] out Should be freed by calling ReleaseValue
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtMemoryInfo)* info,
        void* p_data,
        const(long)* dense_shape,
        size_t dense_shape_len,
        const(long)* values_shape,
        size_t values_shape_len,
        ONNXTensorElementDataType type,
        OrtValue** out_) CreateSparseTensorWithValuesAsOrtValue;

    /**
     * This assigns Coo format indices to the SparseTensor that was created by
     * OrtApi::CreateSparseTensorWithValuesAsOrtValue above. It also sets OrtSparseFormat to
     * ORT_SPARSE_COO. This will not allocate any additional memory for data. The life span of
     * indices_data buffer should eclipse the life span of this ::OrtValue.
     *
     * \param[in,out] ort_value ::OrtValue instance constructed with OrtApi::CreateSparseTensorWithValuesAsOrtValue
     * \param[in,out] indices_data pointer to a user pre-allocated buffer or nullptr for fully sparse tensors.
     * \param[in] indices_num  number of COO indices. Should either be 0 for fully sparse tensors, be equal
     *  to the number of nnz values specified to OrtApi::CreateSparseTensorWithValuesAsOrtValue for 1-D {nnz} indices or
     *  be twice as number of nnz values for a  2-D indices {nnz, 2}
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtValue* ort_value, long* indices_data, size_t indices_num) UseCooIndices;

    /**
     * The assigns CSR format indices to the SparseTensor that was created by
     * OrtApi::CreateSparseTensorWithValuesAsOrtValue above. It also sets OrtSparseFormat to
     * ORT_SPARSE_CSRC. This will not allocate any additional memory for data. The life spans of
     * inner_data and outer_data buffers should eclipse the life span of this ::OrtValue.
     *
     * \param[in,out] ort_value ::OrtValue instance constructed with OrtApi::CreateSparseTensorWithValuesAsOrtValue
     * \param[in,out] inner_data pointer to a user pre-allocated buffer or nullptr for fully sparse tensors.
     * \param[in] inner_num  number of inner CSR indices. Should either be 0 for fully sparse tensors or be equal
     * to the number of nnz values specified to OrtApi::CreateSparseTensorWithValuesAsOrtValue.
     * \param[in,out] outer_data pointer to user pre-allocated buffer or nullptr for fully sparse tensors.
     * \param[in] outer_num number of CSR outer indices. Should either be 0 for fully sparse tensors or
     * equal to rows + 1 of the dense shape.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtValue* ort_value,
        long* inner_data,
        size_t inner_num,
        long* outer_data,
        size_t outer_num) UseCsrIndices;

    /**
     * The assigns BlockSparse format indices to the SparseTensor that was created by
     * OrtApi::CreateSparseTensorWithValuesAsOrtValue above. It also sets OrtSparseFormat to
     * ORT_SPARSE_BLOCK_SPARSE. This will not allocate any additional memory for data. The life span of
     * indices_data buffer must eclipse the lifespan of this ::OrtValue.
     *
     * \param[in,out] ort_value OrtValue instance constructed with OrtApi::CreateSparseTensorWithValuesAsOrtValue
     * \param[in] indices_shape pointer to indices shape. Use {0} for fully sparse tensors
     * \param[in] indices_shape_len length of the indices shape
     * \param[in,out] indices_data pointer to user pre-allocated buffer or nullptr for fully sparse tensors.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtValue* ort_value, const(long)* indices_shape, size_t indices_shape_len, int* indices_data) UseBlockSparseIndices;

    /** \brief Returns sparse tensor format enum iff a given ort value contains an instance of sparse tensor.
     *
     * \param[in] ort_value ::OrtValue that contains an instance of sparse tensor
     * \param[out] out pointer to out parameter
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* ort_value, OrtSparseFormat* out_) GetSparseTensorFormat;

    /** \brief Returns data type and shape of sparse tensor values (nnz) iff ::OrtValue contains a SparseTensor.
     *
     * \param[in] ort_value An ::OrtValue that contains a fully constructed sparse tensor
     * \param[out] out Must be freed by OrtApi::ReleaseTensorTypeAndShapeInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* ort_value, OrtTensorTypeAndShapeInfo** out_) GetSparseTensorValuesTypeAndShape;

    /** \brief Returns numeric data for sparse tensor values (nnz). For string values use GetStringTensor*().
     *
     * \param[in] ort_value an instance of ::OrtValue containing sparse tensor
     * \param[out] out returns a pointer to values data.  Do not attempt to free this ptr.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* ort_value, const(void*)* out_) GetSparseTensorValues;

    /** \brief Returns data type, shape for the type of indices specified by indices_format.
     *
     * \param[in] ort_value ::OrtValue containing sparse tensor.
     * \param[in] indices_format One of the indices formats. It is an error to request a format that the sparse
     * tensor does not contain.
     * \param[out] out an instance of ::OrtTensorTypeAndShapeInfo. Must be freed by OrtApi::ReleaseTensorTypeAndShapeInfo
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* ort_value, OrtSparseIndicesFormat indices_format, OrtTensorTypeAndShapeInfo** out_) GetSparseTensorIndicesTypeShape;

    /** \brief Returns indices data for the type of the indices specified by indices_format
     *
     * \param[in] ort_value ::OrtValue containing sparse tensor.
     * \param[in] indices_format One of the indices formats. It is an error to request a format that the sparse tensor does not contain.
     * \param[out] num_indices Pointer to where the number of indices entries is returned
     * \param[out] indices Returned pointer to the indices data. Do not free the returned pointer as it refers to internal data owned by the ::OrtValue
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* ort_value, OrtSparseIndicesFormat indices_format, size_t* num_indices, const(
            void*)* indices) GetSparseTensorIndices;
    /// @}
    /// \name OrtSessionOptions
    /// @{

    /**
     * \brief Sets out to 1 iff an optional type OrtValue has an element, 0 otherwise (OrtValue is None)
     * Use this API to find if the optional type OrtValue is None or not.
     * If the optional type OrtValue is not None, use the OrtValue just like any other OrtValue.
     * For example, if you get an OrtValue that corresponds to Optional(tensor) and
     * if HasValue() returns true, use it as tensor and so on.

     * \param[in] value Input OrtValue.
     * \param[out] out indicating if the input OrtValue contains data (1) or if it is a None (0)
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, int* out_) HasValue;

    /// @}
    /// \name OrtKernelContext
    /// Custom operator APIs.
    /// @{

    /** \brief Used for custom operators, gets the GPU compute stream to use to launch the custom a GPU kernel
     *   \see ::OrtCustomOp
     * \param[in]  context OrtKernelContext instance
     * \param[out] out Returns pointer to a GPU compute stream that can be used to launch the custom GPU kernel.
     *             If retrieving the GPU compute stream is not relevant (GPU not enabled in the build, kernel partitioned to
     *             some other EP), then a nullptr is returned as the output param.
     *             Do not free or mutate the returned pointer as it refers to internal data owned by the underlying session.
     *             Only use it for custom kernel launching.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtKernelContext)* context, void** out_) KernelContext_GetGPUComputeStream;

    /// @}
    /// \name GetTensorMemoryInfo
    /// @{
    /** \brief Returns a pointer to the ::OrtMemoryInfo of a Tensor
     * \param[in] value ::OrtValue containing tensor.
     * \param[out] mem_info ::OrtMemoryInfo of the tensor. Do NOT free the returned pointer. It is valid for the lifetime of the ::OrtValue
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(OrtValue)* value, const(OrtMemoryInfo*)* mem_info) GetTensorMemoryInfo;

    /// @}
    /// \name GetExecutionProviderApi
    /// @{
    /** \brief Get a pointer to the requested version of the Execution Provider specific
     * API extensions to the OrtApi
     * \param[in] provider_name The name of the execution provider name. Currently only the following
     * values are supported: "DML".
     * \param[in] version Must be ::ORT_API_VERSION.
     * \param[out] provider_api A void pointer containing a reference to the execution provider versioned api structure.
     * For example, the provider_api pointer can be cast to the OrtDmlApi* when the provider_name is "DML".
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(const(char)* provider_name, uint version_, const(void*)* provider_api) GetExecutionProviderApi;

    /// @}

    /// \name SessionOptions
    /// @{
    /** \brief Set custom thread creation function
     *
     * \param[in] options Session options
     * \param[in] ort_custom_create_thread_fn Custom thread creation function
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, OrtCustomCreateThreadFn ort_custom_create_thread_fn) SessionOptionsSetCustomCreateThreadFn;

    /** \brief Set creation options for custom thread
     *
     * \param[in] options Session options
     * \param[in] ort_custom_thread_creation_options Custom thread creation options (can be nullptr)
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, void* ort_custom_thread_creation_options) SessionOptionsSetCustomThreadCreationOptions;

    /** \brief Set custom thread join function
     *
     * \param[in] options Session options
     * \param[in] ort_custom_join_thread_fn Custom join thread function, must not be nullptr when ort_custom_create_thread_fn is set
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtSessionOptions* options, OrtCustomJoinThreadFn ort_custom_join_thread_fn) SessionOptionsSetCustomJoinThreadFn;
    /// @}

    /// \name OrtThreadingOptions
    /// @{
    /** \brief Set custom thread creation function for global thread pools
     *
     * \param[inout] tp_options
     * \param[in] ort_custom_create_thread_fn Custom thread creation function
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtThreadingOptions* tp_options, OrtCustomCreateThreadFn ort_custom_create_thread_fn) SetGlobalCustomCreateThreadFn;

    /** \brief Set custom thread creation options for global thread pools
     *
     * \param[inout] tp_options
     * \param[in] ort_custom_thread_creation_options Custom thread creation options (can be nullptr)
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtThreadingOptions* tp_options, void* ort_custom_thread_creation_options) SetGlobalCustomThreadCreationOptions;

    /** \brief Set custom thread join function for global thread pools
     *
     * \param[inout] tp_options
     * \param[in] ort_custom_join_thread_fn Custom thread join function, must not be nullptr when global ort_custom_create_thread_fn is set
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtThreadingOptions* tp_options, OrtCustomJoinThreadFn ort_custom_join_thread_fn) SetGlobalCustomJoinThreadFn;
    /// @}

    /** \brief Synchronize bound inputs. The call may be necessary for some providers, such as cuda,
     *   in case the system that allocated bound memory operated on a different stream. However, the
     *   operation is provider specific and could be a no-op.
     *
     * \param[inout] binding_ptr
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtIoBinding* binding_ptr) SynchronizeBoundInputs;

    /** \brief Synchronize bound outputs. The call may be necessary for some providers, such as cuda,
     *   in case the system that allocated bound memory operated on a different stream. However, the
     *   operation is provider specific and could be a no-op.
     *
     * \param[inout] binding_ptr
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtIoBinding* binding_ptr) SynchronizeBoundOutputs;

    /// \name OrtSessionOptions
    /// @{

    /** \brief Append CUDA execution provider to the session options
     *
     * If CUDA is not available (due to a non CUDA enabled build), this function will return failure.
     *
     * This is slightly different from OrtApi::SessionOptionsAppendExecutionProvider_CUDA, it takes an
     * ::OrtCUDAProviderOptions which is publicly defined. This takes an opaque ::OrtCUDAProviderOptionsV2
     * which must be created with OrtApi::CreateCUDAProviderOptions.
     *
     * For OrtApi::SessionOptionsAppendExecutionProvider_CUDA, the user needs to instantiate ::OrtCUDAProviderOptions
     * as well as allocate/release buffers for some members of ::OrtCUDAProviderOptions.
     * Here, OrtApi::CreateCUDAProviderOptions and Ortapi::ReleaseCUDAProviderOptions will do the memory management for you.
     *
     * \param[in] options
     * \param[in] cuda_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.11.
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(OrtCUDAProviderOptionsV2)* cuda_options) SessionOptionsAppendExecutionProvider_CUDA_V2;

    /// @}
    /// \name OrtCUDAProviderOptionsV2
    /// @{

    /** \brief Create an OrtCUDAProviderOptionsV2
     *
     * \param[out] out Newly created ::OrtCUDAProviderOptionsV2. Must be released with OrtApi::ReleaseCudaProviderOptions
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.11.
     */
    OrtStatusPtr function(OrtCUDAProviderOptionsV2** out_) CreateCUDAProviderOptions;

    /** \brief Set options in a CUDA Execution Provider.
     *
     * Please refer to https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
     * to know the available keys and values. Key should be in null terminated string format of the member of ::OrtCUDAProviderOptionsV2
     * and value should be its related range.
     *
     * For example, key="device_id" and value="0"
     *
     * \param[in] cuda_options
     * \param[in] provider_options_keys Array of UTF-8 null-terminated string for provider options keys
     * \param[in] provider_options_values Array of UTF-8 null-terminated string for provider options values
     * \param[in] num_keys Number of elements in the `provider_option_keys` and `provider_options_values` arrays
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.11.
     */
    OrtStatusPtr function(
        OrtCUDAProviderOptionsV2* cuda_options,
        const(char*)* provider_options_keys,
        const(char*)* provider_options_values,
        size_t num_keys) UpdateCUDAProviderOptions;

    /**
     * Get serialized CUDA provider options string.
     *
     * For example, "device_id=0;arena_extend_strategy=0;......"
     *
     * \param cuda_options - OrtCUDAProviderOptionsV2 instance
     * \param allocator - a ptr to an instance of OrtAllocator obtained with CreateAllocator() or GetAllocatorWithDefaultOptions()
     *                      the specified allocator will be used to allocate continuous buffers for output strings and lengths.
     * \param ptr - is a UTF-8 null terminated string allocated using 'allocator'. The caller is responsible for using the same allocator to free it.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.11.
     */
    OrtStatusPtr function(const(OrtCUDAProviderOptionsV2)* cuda_options, OrtAllocator* allocator, char** ptr) GetCUDAProviderOptionsAsString;

    /** \brief Release an ::OrtCUDAProviderOptionsV2
     *
     * \note This is an exception in the naming convention of other Release* functions, as the name of the method does not have the V2 suffix, but the type does
     *
     * \since Version 1.11.
     */
    void function(OrtCUDAProviderOptionsV2* input) ReleaseCUDAProviderOptions;

    /// @}

    /** \brief Append MIGraphX provider to session options
     *
     * If MIGraphX is not available (due to a non MIGraphX enabled build, or if MIGraphX is not installed on the system), this function will return failure.
     *
     * \param[in] options
     * \param[in] migraphx_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.11.
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(OrtMIGraphXProviderOptions)* migraphx_options) SessionOptionsAppendExecutionProvider_MIGraphX;

    /** \brief Replace initialized Tensors with external data with the data provided in initializers.
     *
     * The function will find the initialized TensorProtos with external data in the graph with the provided names and
     * replace them with the provided tensors. The API verifies that the TensorProto being replaced
     * has an external data reference and has the same name, dimensions and data type as its replacement. The replacement
     * will occur before any of the optimizations take place. The data will be copied into the graph
     * since TensorProto can't refer to the user provided buffers.
     *
     * Once the model has been loaded, the OrtValue(s) added to SessionOptions instance will be removed
     * from the internal SessionOptions copy to save memory, the user provided buffers can then be deallocated
     * and the SessionOptions instance that refers to them can be destroyed.
     *
     * \param[in] options
     * \param[in] initializer_names Array of null terminated UTF-8 encoded strings of the initializers names.
     * \param[in] initializers Array of ::OrtValue type
     * \param[in] initializers_num Number of elements in the initializer_names and initializers
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.12.
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(char*)* initializer_names,
        const(OrtValue*)* initializers,
        size_t initializers_num) AddExternalInitializers;

    /** \brief: Create attribute of onnxruntime operator
     *
     * \param[in] name Name of the attribute
     * \param[in] data Data content of the attribute
     * \param[in] len Number of bytes stored in data
     * \param[in] type Data type
     * \param[out] op_attr Attribute that has been created, which must be released by OrtApi::ReleaseOpAttr
     *
     * \since Version 1.12.
     */
    OrtStatusPtr function(
        const(char)* name,
        const(void)* data,
        int len,
        OrtOpAttrType type,
        OrtOpAttr** op_attr) CreateOpAttr;

    /* \brief: Release op attribute
     *
     * \param[in] opAttr Attribute created by OrtApi::CreateOpAttr
     *
     * \since Version 1.12.
     */
    void function(OrtOpAttr* input) ReleaseOpAttr;

    /** \brief: Create onnxruntime native operator
     *
     * \param[in] info Kernel info
     * \param[in] op_name Operator name
     * \param[in] domain Operator domain
     * \param[in] version Operator opset version
     * \param[in] type_constraint_names Name of the type contraints, such as "T" or "T1"
     * \param[in] type_constraint_values Type of each contraints
     * \param[in] type_constraint_count Number of contraints
     * \param[in] attr_values Attributes used to initialize the operator
     * \param[in] attr_count Number of the attributes
     * \param[in] input_count Number of inputs
     * \param[in] output_count Number of outputs
     * \param[out] ort_op Operator that has been created
     *
     * \since Version 1.12.
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        const(char)* op_name,
        const(char)* domain,
        int version_,
        const(char*)* type_constraint_names,
        const(ONNXTensorElementDataType)* type_constraint_values,
        int type_constraint_count,
        const(OrtOpAttr*)* attr_values,
        int attr_count,
        int input_count,
        int output_count,
        OrtOp** ort_op) CreateOp;

    /** \brief: Invoke the operator created by OrtApi::CreateOp
     * The inputs must follow the order as specified in onnx specification
     *
     * \param[in] context Kernel context
     * \param[in] ort_op Operator that has been created
     * \param[in] input_values Array of inputs
     * \param[in] input_count Number of inputs
     * \param[in] output_values Array of outputs
     * \param[in] output_count Number of outputs
     *
     * \since Version 1.12.
     */
    OrtStatusPtr function(
        const(OrtKernelContext)* context,
        const(OrtOp)* ort_op,
        const(OrtValue*)* input_values,
        int input_count,
        OrtValue** output_values,
        int output_count) InvokeOp;

    /* \brief: Release an onnxruntime operator
     *
     * \param[in] Op Operator created by OrtApi::CreateOp
     *
     * \since Version 1.12.
     */
    void function(OrtOp* input) ReleaseOp;

    /** \brief: Append execution provider to the session options.
     * \param[in] options
     * \param[in] provider_name - provider to add.
     * \param[in] provider_options_keys - keys to configure the provider options
     * \param[in] provider_options_values - values to configure the provider options
     * \param[in] num_keys - number of keys passed in
     *
     * Currently supported providers:
     *   QNN
     *   SNPE
     *   XNNPACK
     *
     * Note: If an execution provider has a dedicated SessionOptionsAppendExecutionProvider_<provider name> function
     *       that should be used to add it.
     *
     * QNN supported keys:
     *   "backend_path": file path to QNN backend library.
     *   "profiling_level": QNN profiling level, options: "off", "basic", "detailed". Default to off.
     *   "rpc_control_latency": QNN RPC control latency.
     *   "vtcm_mb": QNN VTCM size in MB. default to 0(not set).
     *   "htp_performance_mode": QNN performance mode, options: "burst", "balanced", "default", "high_performance",
     *   "high_power_saver", "low_balanced", "extreme_power_saver", "low_power_saver", "power_saver", "sustained_high_performance". Default to "default".
     *   "qnn_saver_path": File path to the QNN Saver backend library. If specified, QNN Saver will be enabled and will
     *   dump QNN API calls to disk for replay/debugging. QNN Saver produces incorrect model inference results and
     *   may alter model/EP partitioning. Use only for debugging.
     *   "qnn_context_priority": QNN context priority, options: "low", "normal", "normal_high", "high". Default to "normal".
     *   "htp_graph_finalization_optimization_mode": Set the optimization mode for graph finalization on the HTP backend. Available options:
     *     - "0": Default.
     *     - "1": Faster preparation time, less optimal graph.
     *     - "2": Longer preparation time, more optimal graph.
     *     - "3": Longest preparation time, most likely even more optimal graph. See QNN SDK documentation for specific details.
     *   "soc_model": The SoC model number. Refer to the QNN SDK documentation for valid values. Defaults to "0" (unknown).
     *   "htp_arch": The minimum HTP architecture the driver will use to select compatible QNN operators. Available options:
     *     - "0": Default (none).
     *     - "68"
     *     - "69"
     *     - "73"
     *     - "75"
     *   "device_id": The ID of the device to use when setting 'htp_arch'. Defaults to "0" (for single device).
     *
     * SNPE supported keys:
     *   "runtime": SNPE runtime engine, options: "CPU", "CPU_FLOAT32", "GPU", "GPU_FLOAT32_16_HYBRID", "GPU_FLOAT16",
     *   "DSP", "DSP_FIXED8_TF", "AIP_FIXED_TF", "AIP_FIXED8_TF".
     *   Mapping to SNPE Runtime_t definition: CPU, CPU_FLOAT32 => zdl::DlSystem::Runtime_t::CPU;
     *   GPU, GPU_FLOAT32_16_HYBRID => zdl::DlSystem::Runtime_t::GPU;
     *   GPU_FLOAT16 => zdl::DlSystem::Runtime_t::GPU_FLOAT16;
     *   DSP, DSP_FIXED8_TF => zdl::DlSystem::Runtime_t::DSP.
     *   AIP_FIXED_TF, AIP_FIXED8_TF => zdl::DlSystem::Runtime_t::AIP_FIXED_TF.
     *   "priority": execution priority, options: "low", "normal".
     *   "buffer_type": ITensor or user buffers, options: "ITENSOR", user buffer with different types - "TF8", "TF16", "UINT8", "FLOAT".
     *   "ITENSOR" -- default, ITensor which is float only.
     *   "TF8" -- quantized model required, "FLOAT" -- for both quantized or non-quantized model
     *   "enable_init_cache": enable SNPE init caching feature, set to 1 to enabled it. Disabled by default.
     *   If SNPE is not available (due to a non Snpe enabled build or its dependencies not being installed), this function will fail.
     *
     * XNNPACK supported keys:
     *   "intra_op_num_threads": number of thread-pool size to use for XNNPACK execution provider.
     *      default value is 0, which means to use the session thread-pool size.
     *
     * \since Version 1.12.
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(char)* provider_name,
        const(char*)* provider_options_keys,
        const(char*)* provider_options_values,
        size_t num_keys) SessionOptionsAppendExecutionProvider;

    /* \brief: Get a copy of kernel info
     *
     * \param[in] info Kernel info
     * \param[out] info_copy Copy of kernel info
     *
     * \since Version 1.12.
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        OrtKernelInfo** info_copy) CopyKernelInfo;

    /* \brief: Release kernel info
     *
     * \param[in] KernelInfo A copy of kernel info returned by CopyKernelInfo
     *
     * \since Version 1.12.
     */
    void function(OrtKernelInfo* input) ReleaseKernelInfo;

    /// \name Ort Training
    /// @{
    /** \brief Gets the Training C Api struct
     *
     * Call this function to access the ::OrtTrainingApi structure that holds pointers to functions that enable
     * training with onnxruntime.
     * \note A NULL pointer will be returned and no error message will be printed if the training api
     * is not supported with this build. A NULL pointer will be returned and an error message will be
     * printed if the provided version is unsupported, for example when using a runtime older than the
     * version created with this header file.
     *
     * \param[in] version Must be ::ORT_API_VERSION
     * \return The ::OrtTrainingApi struct for the version requested.
     *
     * \since Version 1.13
     */
    const(OrtTrainingApi)* function(uint version_) GetTrainingApi;

    /// @}

    /** \brief Append CANN provider to session options
     *
     * If CANN is not available (due to a non CANN enabled build, or if CANN is not installed on the system), this function will return failure.
     *
     * \param[in] options
     * \param[in] cann_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.13.
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(OrtCANNProviderOptions)* cann_options) SessionOptionsAppendExecutionProvider_CANN;

    /** \brief Create an OrtCANNProviderOptions
     *
     * \param[out] out created ::OrtCANNProviderOptions. Must be released with OrtApi::ReleaseCANNProviderOptions
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.13.
     */
    OrtStatusPtr function(OrtCANNProviderOptions** out_) CreateCANNProviderOptions;

    /** \brief Set options in a CANN Execution Provider.
     *
     * \param[in] cann_options
     * \param[in] provider_options_keys Array of UTF-8 null-terminated string for provider options keys
     * \param[in] provider_options_values Array of UTF-8 null-terminated string for provider options values
     * \param[in] num_keys Number of elements in the `provider_option_keys` and `provider_options_values` arrays
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.13.
     */
    OrtStatusPtr function(
        OrtCANNProviderOptions* cann_options,
        const(char*)* provider_options_keys,
        const(char*)* provider_options_values,
        size_t num_keys) UpdateCANNProviderOptions;

    /** \brief Get serialized CANN provider options string.
     *
     * \param[in] cann_options OrtCANNProviderOptions instance
     * \param[in] allocator a ptr to an instance of OrtAllocator obtained with CreateAllocator()
     *                      or GetAllocatorWithDefaultOptions(), the specified allocator will be used to allocate
     *                      continuous buffers for output strings and lengths.
     * \param[out] ptr is a UTF-8 null terminated string allocated using 'allocator'.
     *                 The caller is responsible for using the same allocator to free it.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.13.
     */
    OrtStatusPtr function(
        const(OrtCANNProviderOptions)* cann_options,
        OrtAllocator* allocator,
        char** ptr) GetCANNProviderOptionsAsString;

    /** \brief Release an OrtCANNProviderOptions
     *
     * \param[in] the pointer of OrtCANNProviderOptions which will been deleted
     *
     * \since Version 1.13.
     */
    void function(OrtCANNProviderOptions* input) ReleaseCANNProviderOptions;

    /*  \brief Get OrtDevice type from MemoryInfo
     *
     *  \since Version 1.14
     */
    void function(const(OrtMemoryInfo)* ptr, OrtMemoryInfoDeviceType* out_) MemoryInfoGetDeviceType;

    /* \brief Update the OrtEnv instance with custom log severity level
     *
     * \param[in] ort_env The OrtEnv instance being used
     * \param[in] log_severity_level The log severity level.
     *
     * \since Version 1.14.
     */
    OrtStatusPtr function(OrtEnv* ort_env, OrtLoggingLevel log_severity_level) UpdateEnvWithCustomLogLevel;

    /*  \brief Set affinities for intra op threads
     *
     * Affinity string follows format:
     * logical_processor_id,logical_processor_id;logical_processor_id,logical_processor_id
     * Semicolon isolates configurations among threads, while comma split processors where ith thread expected to attach to.
     * e.g. 1,2,3;4,5
     * specifies affinities for two threads, with the 1st thread attach to the 1st, 2nd, and 3rd processor, and 2nd thread to the 4th and 5th.
     * To ease the configuration, an "interval" is also allowed:
     * e.g. 1-8;8-16;17-24
     * orders that the 1st thread runs on first eight processors, 2nd thread runs on next eight processors, and so forth.
     * Note:
     * 1. Once set, the number of thread affinities must equal to intra_op_num_threads - 1,
     *    ort does not set affinity on the main thread which is started and managed by the calling app;
     * 2. For windows, ort will infer the group id from a logical processor id, for example, assuming there are two groups with each has 64 logical processors,
     *    an id of 64 will be inferred as the last processor of the 1st group, while 65 will be interpreted as the 1st processor of the second group.
     *    Hence 64-65 is an invalid configuration, because a windows thread cannot be attached to processors across group boundary.
     *
     *  \since Version 1.14
     */
    OrtStatusPtr function(OrtThreadingOptions* tp_options, const(char)* affinity_string) SetGlobalIntraOpThreadAffinity;

    /** \brief Register custom ops from a shared library.
     *
     * Loads a shared library (.dll on windows, .so on linux, etc) named 'library_name' and looks for this entry point:
     *		OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
     * It then passes in the provided session options to this function along with the api base.
     *
     * The handle to the loaded library is automatically released by ORT when the last OrtSession that references the
     * library handle is released. If no OrtSession is created, then the library handle is released when the provided
     * OrtSessionOptions is released.
     *
     * \param[in] options The session options.
     * \param[in] library_name The name of the shared library to load and register. Refer to OS-specific dynamic library
     *                         loading utilities (e.g., LoadLibraryEx on Windows or dlopen on Linux/MacOS) for information
     *                         on the format of library names and search paths.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(OrtSessionOptions* options, const(char)* library_name) RegisterCustomOpsLibrary_V2;

    /** \brief Register custom ops by calling a RegisterCustomOpsFn function.
     *
     * Searches for registration_func_name and if found calls it.
     *
     * The library containing the function must either be linked against or previously loaded by the executable.
     *
     * If you want ONNX Runtime to load the library and manage its lifetime, use RegisterCustomOpsLibrary_V2.
     *
     * RegisterCustomOpsUsingFunction can be used in scenarios where it may not be possible for ONNX Runtime to load
     * the library from a path. e.g. mobile platforms where the library must be linked into the app.
     *
     * The registration function must have the signature of RegisterCustomOpsFn:
     *    OrtStatus* (*fn)(OrtSessionOptions* options, const OrtApiBase* api);
     *
     * See https://onnxruntime.ai/docs/reference/operators/add-custom-op.html for details on how the registration
     * function should be implemented.
     *
     * \param[in] options OrtSessionOptions that is passed through as the first argument in the call to the
     *                    registration function.
     * \param[in] registration_func_name Name of registration function to use.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(char)* registration_func_name) RegisterCustomOpsUsingFunction;

    /// \name OrtKernelInfo
    /// Custom operator APIs.
    /// @{

    /** \brief Get the number of inputs from ::OrtKernelInfo.
     *
     * Used in the CreateKernel callback of an OrtCustomOp to query the number of inputs
     * during kernel/session creation.
     *
     * \param[in] info Instance of ::OrtKernelInfo.
     * \param[out] out Pointer to variable assigned with the result on success.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(const(OrtKernelInfo)* info, size_t* out_) KernelInfo_GetInputCount;

    /** \brief Get the number of outputs from ::OrtKernelInfo.
     *
     * Used in the CreateKernel callback of an OrtCustomOp to query the number of outputs
     * during kernel/session creation.
     *
     * \param[in] info Instance of ::OrtKernelInfo.
     * \param[out] out Pointer to variable assigned with the result on success.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(const(OrtKernelInfo)* info, size_t* out_) KernelInfo_GetOutputCount;

    /** \brief Get the name of a ::OrtKernelInfo's input.
     *
     * Used in the CreateKernel callback of an OrtCustomOp to query an input's name
     * during kernel/session creation.
     *
     * If `out` is nullptr, the value of `size` is set to the size of the name
     * string (including null-terminator), and a success status is returned.
     *
     * If the `size` parameter is greater than or equal to the name string's size,
     * the value of `size` is set to the true size of the string (including null-terminator),
     * the provided memory is filled with the string's contents, and a success status is returned.
     *
     * If the `size` parameter is less than the actual string's size and `out`
     * is not nullptr, the value of `size` is set to the true size of the string
     * and a failure status is returned.
     *
     * \param[in] info An instance of ::OrtKernelInfo.
     * \param[in] index The index of the input name to get. Returns a failure status if out-of-bounds.
     * \param[out] out Memory location into which to write the UTF-8 null-terminated string representing the input's name.
     * \param[in,out] size Pointer to the size of the `out` buffer. See above comments for details.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        size_t index,
        char* out_,
        size_t* size) KernelInfo_GetInputName;

    /** \brief Get the name of a ::OrtKernelInfo's output.
     *
     * Used in the CreateKernel callback of an OrtCustomOp to query an output's name
     * during kernel/session creation.
     *
     * If `out` is nullptr, the value of `size` is set to the size of the name
     * string (including null-terminator), and a success status is returned.
     *
     * If the `size` parameter is greater than or equal to the name string's size,
     * the value of `size` is set to the true size of the string (including null-terminator),
     * the provided memory is filled with the string's contents, and a success status is returned.
     *
     * If the `size` parameter is less than the actual string's size and `out`
     * is not nullptr, the value of `size` is set to the true size of the string
     * and a failure status is returned.
     *
     * \param[in] info An instance of ::OrtKernelInfo.
     * \param[in] index The index of the output name to get. Returns a failure status if out-of-bounds.
     * \param[out] out Memory location into which to write the UTF-8 null-terminated string representing the output's
     *                 name.
     * \param[in,out] size Pointer to the size of the `out` buffer. See above comments for details.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        size_t index,
        char* out_,
        size_t* size) KernelInfo_GetOutputName;

    /** \brief Get the type information for a ::OrtKernelInfo's input.
     *
     * Used in the CreateKernel callback of an OrtCustomOp to query the shape and type information
     * of an input during kernel/session creation.
     *
     * \param[in] info An instance of ::OrtKernelInfo.
     * \param[in] index Which input to get the type information for
     * \param[out] type_info Pointer set to the resulting ::OrtTypeInfo. Must be freed with OrtApi::ReleaseTypeInfo.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        size_t index,
        OrtTypeInfo** type_info) KernelInfo_GetInputTypeInfo;

    /** \brief Get the type information for a ::OrtKernelInfo's output.
     *
     * Used in the CreateKernel callback of an OrtCustomOp to query the shape and type information
     * of an output during kernel/session creation.
     *
     * \param[in] info An instance of ::OrtKernelInfo.
     * \param[in] index Which input to get the type information for
     * \param[out] type_info Pointer set to the resulting ::OrtTypeInfo. Must be freed with OrtApi::ReleaseTypeInfo.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        size_t index,
        OrtTypeInfo** type_info) KernelInfo_GetOutputTypeInfo;

    /** \brief Get a ::OrtValue tensor stored as an attribute in the graph node.
     *
     * Used in the CreateKernel callback of an OrtCustomOp to get a tensor attribute.
     *
     * \param[in] info ::OrtKernelInfo instance.
     * \param[in] name UTF-8 null-terminated string representing the attribute's name.
     * \param[in] allocator Allocator used to allocate the internal tensor state.
     * \param[out] out Returns newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue,
     *                 which will also free internal tensor state allocated with the provided allocator.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        const(OrtKernelInfo)* info,
        const(char)* name,
        OrtAllocator* allocator,
        OrtValue** out_) KernelInfoGetAttribute_tensor;

    /// @}
    /// \name OrtSessionOptions
    /// Custom operator APIs
    /// @{

    /** \brief Checks if the given session configuration entry exists.
     *
     * The config_key formats are defined in onnxruntime_session_options_config_keys.h
     *
     * Can be used in a custom operator library to check for session configuration entries
     * that target one or more custom operators in the library. Example: The config entry
     * custom_op.myop.some_key targets a custom op named "myop".
     *
     * \param[in] options The ::OrtSessionOptions instance.
     * \param[in] config_key A null-terminated UTF-8 string representation of the configuration key.
     * \param[out] out Pointer set to 1 if the entry exists and 0 otherwise.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(
        const(OrtSessionOptions)* options,
        const(char)* config_key,
        int* out_) HasSessionConfigEntry;

    /** \brief Get a session configuration value.
     *
     * Returns a failure status if the configuration key does not exist.
     * The config_key and the format of config_value are defined in onnxruntime_session_options_config_keys.h
     *
     * If `config_value` is nullptr, the value of `size` is set to the true size of the string
     * value (including null-terminator), and a success status is returned.
     *
     * If the `size` parameter is greater than or equal to the actual string value's size,
     * the value of `size` is set to the true size of the string value, the provided memory
     * is filled with the value's contents, and a success status is returned.
     *
     * If the `size` parameter is less than the actual string value's size and `config_value`
     * is not nullptr, the value of `size` is set to the true size of the string value
     * and a failure status is returned.
     *
     * Can be used in a custom operator library to get session configuration entries
     * that target one or more custom operators in the library. Example: The config entry
     * custom_op.myop.some_key targets a custom op named "myop".
     *
     * \param[in] options The session options.
     * \param[in] config_key A null-terminated UTF-8 string representation of the config key.
     * \param[in] config_value Pointer to memory where the null-terminated UTF-8 string value will be stored.
     * \param[in,out] size Pointer to the size of the `config_value` buffer. See above comments for details.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.14
     */
    OrtStatusPtr function(
        const(OrtSessionOptions)* options,
        const(char)* config_key,
        char* config_value,
        size_t* size) GetSessionConfigEntry;

    /// @}

    /** \brief Append dnnl provider to session options
     *
     * If oneDNN is not available, this function will return failure.
     *
     * \param[in] options
     * \param[in] dnnl_options
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.15.
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(OrtDnnlProviderOptions)* dnnl_options) SessionOptionsAppendExecutionProvider_Dnnl;

    /** \brief Create an OrtDnnlProviderOptions
     *
     * \param[out] out Newly created ::OrtDnnlProviderOptions. Must be released with OrtApi::ReleaseDnnlProviderOptions
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.15.
     */
    OrtStatusPtr function(OrtDnnlProviderOptions** out_) CreateDnnlProviderOptions;

    /** \brief Set options in a oneDNN Execution Provider.
     *
     * Key should be in null terminated string format of the member of ::OrtDnnlProviderOptions
     * and value should be its related range.
     *
     * For example, key="use_arena" and value="1"
     *
     * \param[in] dnnl_options
     * \param[in] provider_options_keys Array of UTF-8 null-terminated string for provider options keys
     * \param[in] provider_options_values Array of UTF-8 null-terminated string for provider options values
     * \param[in] num_keys Number of elements in the `provider_option_keys` and `provider_options_values` arrays
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.15.
     */
    OrtStatusPtr function(
        OrtDnnlProviderOptions* dnnl_options,
        const(char*)* provider_options_keys,
        const(char*)* provider_options_values,
        size_t num_keys) UpdateDnnlProviderOptions;

    /**
     * Get serialized oneDNN provider options string.
     *
     * For example, "use_arena=1;......"
     *
     * \param dnnl_options - OrtDnnlProviderOptions instance
     * \param allocator - a ptr to an instance of OrtAllocator obtained with CreateAllocator() or GetAllocatorWithDefaultOptions()
     *                      the specified allocator will be used to allocate continuous buffers for output strings and lengths.
     * \param ptr - is a UTF-8 null terminated string allocated using 'allocator'. The caller is responsible for using the same allocator to free it.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.15.
     */
    OrtStatusPtr function(const(OrtDnnlProviderOptions)* dnnl_options, OrtAllocator* allocator, char** ptr) GetDnnlProviderOptionsAsString;

    /** \brief Release an ::OrtDnnlProviderOptions
     *
     * \since Version 1.15.
     */
    void function(OrtDnnlProviderOptions* input) ReleaseDnnlProviderOptions;

    /// \name OrtKernelInfo
    /// Custom operator APIs.
    /// @{

    /** \brief Get the graph node name from ::OrtKernelInfo.
     *
     * If `out` is nullptr, the value of `size` is set to the size of the name
     * string (including null-terminator), and a success status is returned.
     *
     * If the `size` parameter is greater than or equal to the name string's size,
     * the value of `size` is set to the true size of the string (including null-terminator),
     * the provided memory is filled with the string's contents, and a success status is returned.
     *
     * If the `size` parameter is less than the actual string's size and `out`
     * is not nullptr, the value of `size` is set to the true size of the string
     * and a failure status is returned.
     *
     * Can be used in a custom operator's CreateKernel callback to get the name of the operator's node name in the graph.
     *
     * \param[in] info An instance of ::OrtKernelInfo.
     * \param[out] out Memory location into which to write the UTF-8 null-terminated string representing the name.
     * \param[in,out] size Pointer to the size of the `out` buffer. See above comments for details.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.15
     */
    OrtStatusPtr function(const(OrtKernelInfo)* info, char* out_, size_t* size) KernelInfo_GetNodeName;

    /** \brief Get the session logger from ::OrtKernelInfo.
     *
     * Used in the CreateKernel callback of an OrtCustomOp to get a logger that can be used to log
     * messages.
     *
     * \param[in] info An instance of ::OrtKernelInfo.
     * \param[out] logger Pointer set to the session's ::OrtLogger. Owned by ONNX Runtime, so do not free.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.15
     */
    OrtStatusPtr function(const(OrtKernelInfo)* info, const(OrtLogger*)* logger) KernelInfo_GetLogger;

    /// @}
    /// \name OrtKernelContext
    /// Custom operator APIs.
    /// @{

    /** \brief Get the runtime logger from ::OrtKernelContext.
     *
     * Used in the KernelCompute callback of an OrtCustomOp to get a logger that can be used to log
     * messages during inference.
     *
     * \param[in] context An instance of ::OrtKernelContext.
     * \param[out] logger Pointer set to the kernel context's ::OrtLogger. Owned by ONNX Runtime, so do not free.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.15
     */
    OrtStatusPtr function(const(OrtKernelContext)* context, const(OrtLogger*)* logger) KernelContext_GetLogger;

    /// @}
    /// \name OrtLogger
    /// Custom operator APIs.
    /// @{

    /** \brief Logs a message at the given severity level using the provided ::OrtLogger.
     *
     * Only messages with a severity level equal or greater than the ::OrtLogger's logging severity level
     * are logged. Use OrtApi::Logger_GetLoggingSeverityLevel to get the ::OrtLogger's logging severity
     * level.
     *
     * Can be used in custom operators to log messages with the logger retrieved via OrtApi::KernelInfo_GetLogger.
     *
     * \param[in] logger The ::OrtLogger instance.
     * \param[in] log_severity_level The message's severity level.
     * \param[in] message The message to log.
     * \param[in] file_path The filepath of the file in which the message is logged. Usually the value of ORT_FILE.
     * \param[in] line_number The file line number in which the message is logged. Usually the value of __LINE__.
     * \param[in] func_name The name of the function in which the message is logged. Usually the value of __FUNCTION__.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.15
     */
    OrtStatusPtr function(
        const(OrtLogger)* logger,
        OrtLoggingLevel log_severity_level,
        const(char)* message,
        const(char)* file_path,
        int line_number,
        const(char)* func_name) Logger_LogMessage;

    /** \brief Get the logging severity level of the ::OrtLogger.
     *
     * Can be used in a custom operator to get the logging serverity level of the ::OrtLogger associated with
     * the ::OrtKernelInfo.
     *
     * \param[in] logger The ::OrtLogger instance.
     * \param[out] out Pointer to variable assigned with the logging severity level on success.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     * \since Version 1.15
     */
    OrtStatusPtr function(const(OrtLogger)* logger, OrtLoggingLevel* out_) Logger_GetLoggingSeverityLevel;

    /// @}

    /** \brief Get a ::OrtValue tensor stored as a constant initializer in the graph node.
     *
     * Used in the CreateKernel callback of an OrtCustomOp to get a tensor value.
     *
     * \param[in] info ::OrtKernelInfo instance.
     * \param[in] index The node index.
     * \param[out] is_constant Is it a constant node input or not.
     * \param[out] out The OrtValue tensor value.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.15.
     */
    OrtStatusPtr function(const(OrtKernelInfo)* info, size_t index, int* is_constant, const(
            OrtValue*)* out_) KernelInfoGetConstantInput_tensor;

    /** \brief Get Optional Type information from an ::OrtTypeInfo
     *
     * This augments ::OrtTypeInfo to return an ::OrtOptionalTypeInfo when the type is optional.
     * The OrtOptionalTypeInfo also has a nested ::OrtTypeInfo that describes the type of the optional value.
     * ::OrtOptionalTypeInfo type can only appear within model metadata to describe inputs/outputs.
     * The actual OrtValues that are supplied in place of optional type inputs should contain
     * specific type that is described by ::OrtOptionalTypeInfo.
     *
     * So the picture: ::OrtTypeInfo -> ::OrtOptionalTypeInfo -> ::OrtTypeInfo (describes the type that can be supplied
     * in place of the optional type when creating the actual ::OrtValue).
     *
     * \param[in] type_info
     * \param[out] out A pointer to the ::OrtOptionalTypeInfo. Do not free this value,
     *                 it is owned by OrtTypeInfo instance. When the type_info does not represent
     *                 optional type, nullptr is returned in out.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.15.
     */
    OrtStatusPtr function(
        const(OrtTypeInfo)* type_info,
        const(OrtOptionalTypeInfo*)* out_) CastTypeInfoToOptionalTypeInfo;

    /** \brief Get OrtTypeInfo for the allowed contained type from an ::OrtOptionalTypeInfo.
     *
     * This augments ::OrtOptionalTypeInfo to return an ::OrtTypeInfo for the contained type.
     * The OrtOptionalTypeInfo has a nested ::OrtTypeInfo that describes the type of the optional value.
     * ::OrtOptionalTypeInfo type can only appear within model metadata to describe inputs/outputs.
     * The actual OrtValues that are supplied in place of optional type inputs should contain
     * specific type that is described by the returned ::OrtTypeInfo.
     *
     * \param[in] optional_type_info
     * \param[out] out A pointer to the ::OrtTypeInfo for what the optional value could be.
     * it is owned by OrtOptionalTypeInfo instance.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.15.
     */
    OrtStatusPtr function(
        const(OrtOptionalTypeInfo)* optional_type_info,
        OrtTypeInfo** out_) GetOptionalContainedTypeInfo;

    /** \brief Set a single string in a string tensor
     *  Do not zero terminate the string data.
     *
     * \param[in] value A string tensor
     * \param[in] index - flat index of the element
     * \param[in] length_in_bytes length of the buffer in utf-8 bytes (without the null terminator)
     * \param[inout] buffer - address of return value
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(OrtValue* value, size_t index, size_t length_in_bytes, char** buffer) GetResizedStringTensorElementBuffer;

    /** \brief Get Allocator from KernelContext for a specific memoryInfo. Please use C API ReleaseAllocator to release out object
     *
     * \param[in] context OrtKernelContext instance
     * \param[in] mem_info OrtMemoryInfo instance
     * \param[out] out A pointer to OrtAllocator.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.15.
     */
    OrtStatusPtr function(const(OrtKernelContext)* context, const(OrtMemoryInfo)* mem_info, OrtAllocator** out_) KernelContext_GetAllocator;

    /** \brief Returns a null terminated string of the build info including git info and cxx flags
     *
     * \return UTF-8 encoded version string. Do not deallocate the returned buffer.
     *
     * \since Version 1.15.
     */
    const(char)* function() GetBuildInfoString;

    /// \name OrtROCMProviderOptions
    /// @{

    /** \brief Create an OrtROCMProviderOptions
     *
     * \param[out] out Newly created ::OrtROCMProviderOptions. Must be released with OrtApi::ReleaseROCMProviderOptions
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.16.
     */
    OrtStatusPtr function(OrtROCMProviderOptions** out_) CreateROCMProviderOptions;

    /** \brief Set options in a ROCm Execution Provider.
     *
     * Please refer to https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html
     * to know the available keys and values. Key should be in null terminated string format of the member of
     * ::OrtROCMProviderOptions and value should be its related range.
     *
     * For example, key="device_id" and value="0"
     *
     * \param[in] rocm_options
     * \param[in] provider_options_keys Array of UTF-8 null-terminated string for provider options keys
     * \param[in] provider_options_values Array of UTF-8 null-terminated string for provider options values
     * \param[in] num_keys Number of elements in the `provider_option_keys` and `provider_options_values` arrays
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.16.
     */
    OrtStatusPtr function(
        OrtROCMProviderOptions* rocm_options,
        const(char*)* provider_options_keys,
        const(char*)* provider_options_values,
        size_t num_keys) UpdateROCMProviderOptions;

    /**
     * Get serialized ROCm provider options string.
     *
     * For example, "device_id=0;arena_extend_strategy=0;......"
     *
     * \param rocm_options - OrtROCMProviderOptions instance
     * \param allocator - a ptr to an instance of OrtAllocator obtained with CreateAllocator() or GetAllocatorWithDefaultOptions()
     *                      the specified allocator will be used to allocate continuous buffers for output strings and lengths.
     * \param ptr - is a UTF-8 null terminated string allocated using 'allocator'. The caller is responsible for using the same allocator to free it.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.16.
     */
    OrtStatusPtr function(const(OrtROCMProviderOptions)* rocm_options, OrtAllocator* allocator, char** ptr) GetROCMProviderOptionsAsString;

    /** \brief Release an ::OrtROCMProviderOptions
     *
     * \note This is an exception in the naming convention of other Release* functions, as the name of the method does not have the V2 suffix, but the type does
     *
     * \since Version 1.16.
     */
    void function(OrtROCMProviderOptions* input) ReleaseROCMProviderOptions;

    /** \brief Create an allocator with specific type and register it with the ::OrtEnv
     *  This API enhance CreateAndRegisterAllocator that it can create an allocator with specific type, not just CPU allocator
     *  Enables sharing the allocator between multiple sessions that use the same env instance.
     *  Lifetime of the created allocator will be valid for the duration of the environment.
     *  Returns an error if an allocator with the same ::OrtMemoryInfo is already registered.
     *  \param[in] env OrtEnv instance
     *  \param[in] provider_type ExecutionProvider type
     *  \param[in] mem_info OrtMemoryInfo instance
     *  \param[in] arena_cfg Arena configuration
     *  \param[in] provider_options_keys key of the provider options map
     *  \param[in] provider_options_values value of the provider options map
     *  \param[in] num_keys Length of the provider options map
     */
    OrtStatusPtr function(
        OrtEnv* env,
        const(char)* provider_type,
        const(OrtMemoryInfo)* mem_info,
        const(OrtArenaCfg)* arena_cfg,
        const(char*)* provider_options_keys,
        const(char*)* provider_options_values,
        size_t num_keys) CreateAndRegisterAllocatorV2;

    /** \brief Run the model asynchronously in a thread owned by intra op thread pool
     *
     * \param[in] session
     * \param[in] run_options If nullptr, will use a default ::OrtRunOptions
     * \param[in] input_names Array of null terminated UTF8 encoded strings of the input names
     * \param[in] input Array of ::OrtValue%s of the input values
     * \param[in] input_len Number of elements in the input_names and inputs arrays
     * \param[in] output_names Array of null terminated UTF8 encoded strings of the output names
     * \param[in] output_names_len Number of elements in the output_names and outputs array
     * \param[out] output OrtValue* array of size output_names_len.
     *             On calling RunAsync, output[i] could either be a null or a pointer to a preallocated OrtValue.
     *             Later, the output array will be passed to run_async_callback with all null(s) filled with valid
     *             OrtValue pointer(s) allocated by onnxruntime.
     *             NOTE: it is customer's duty to finally release the output array and each of its member,
     *             regardless of whether the member (OrtValue*) is allocated by onnxruntime or preallocated by the customer.
     * \param[in] run_async_callback Callback function on model run completion
     * \param[in] user_data User data that pass back to run_async_callback
     */
    OrtStatusPtr function(
        OrtSession* session,
        const(OrtRunOptions)* run_options,
        const(char*)* input_names,
        const(OrtValue*)* input,
        size_t input_len,
        const(char*)* output_names,
        size_t output_names_len,
        OrtValue** output,
        RunAsyncCallbackFn run_async_callback,
        void* user_data) RunAsync;

    /**
     * Update TensorRT EP provider option where its data type is pointer, for example 'user_compute_stream'.
     * If the data type of the provider option can be represented by string please use UpdateTensorRTProviderOptions.
     *
     * Note: It's caller's responsibility to properly manage the lifetime of the instance pointed by this pointer.
     *
     * \param tensorrt_options - OrtTensorRTProviderOptionsV2 instance
     * \param key - Name of the provider option
     * \param value - A pointer to the instance that will be assigned to this provider option
     *
     * \since Version 1.16.
     */
    OrtStatusPtr function(OrtTensorRTProviderOptionsV2* tensorrt_options, const(char)* key, void* value) UpdateTensorRTProviderOptionsWithValue;

    /**
     * Get TensorRT EP provider option where its data type is pointer.
     * If the data type of the provider option can be represented by string please use GetTensorRTProviderOptionsAsString.
     *
     * \param tensorrt_options - OrtTensorRTProviderOptionsV2 instance
     * \param key - Name of the provider option
     * \param ptr - A pointer to the instance that is kept by the provider option
     *
     * \since Version 1.16.
     */
    OrtStatusPtr function(const(OrtTensorRTProviderOptionsV2)* tensorrt_options, const(char)* key, void** ptr) GetTensorRTProviderOptionsByName;

    /**
     * Update CUDA EP provider option where its data type is pointer, for example 'user_compute_stream'.
     * If the data type of the provider option can be represented by string please use UpdateCUDAProviderOptions.
     *
     * Note: It's caller's responsibility to properly manage the lifetime of the instance pointed by this pointer.
     *
     * \param cuda_options - OrtCUDAProviderOptionsV2 instance
     * \param key - Name of the provider option
     * \param value - A pointer to the instance that will be assigned to this provider option
     *
     * \since Version 1.16.
     */
    OrtStatusPtr function(OrtCUDAProviderOptionsV2* cuda_options, const(char)* key, void* value) UpdateCUDAProviderOptionsWithValue;

    /**
     * Get CUDA EP provider option where its data type is pointer.
     * If the data type of the provider option can be represented by string please use GetCUDAProviderOptionsAsString.
     *
     * \param cuda_options - OrtCUDAProviderOptionsV2 instance
     * \param key - Name of the provider option
     * \param ptr - A pointer to the instance that is kept by the provider option
     *
     * \since Version 1.16.
     */
    OrtStatusPtr function(const(OrtCUDAProviderOptionsV2)* cuda_options, const(char)* key, void** ptr) GetCUDAProviderOptionsByName;

    /**
     * Get a EP resource.
     * E.g. a cuda stream or a cublas handle
     *
     * \param context - Kernel context
     * \param resouce_version - Version of the resource
     * \param resource_id - Type of resource
     * \param resource - A pointer to returned resource
     *
     * \since Version 1.16.
     */
    OrtStatusPtr function(const(OrtKernelContext)* context, int resouce_version, int resource_id, void** resource) KernelContext_GetResource;

    /** \brief Set user logging function
     *
     *  By default the logger created by the CreateEnv* functions is used to create the session logger as well.
     *  This function allows a user to override this default session logger with a logger of their own choosing. This way
     *  the user doesn't have to create a separate environment with a custom logger. This addresses the problem when
     *  the user already created an env but now wants to use a different logger for a specific session (for debugging or
     *  other reasons).
     *
     * \param[in] options
     * \param[in] user_logging_function A pointer to a logging function.
     * \param[in] user_logging_param A pointer to arbitrary data passed as the ::OrtLoggingFunction `param` parameter to
     *                         `user_logging_function`. This parameter is optional.
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     *
     * \since Version 1.17.
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        OrtLoggingFunction user_logging_function,
        void* user_logging_param) SetUserLoggingFunction;

    /**
     * Get number of input from OrtShapeInferContext
     *
     * \param[in] context
     * \param[out] out The number of inputs
     *
     * \since Version 1.17.
     */
    OrtStatusPtr function(const(OrtShapeInferContext)* context, size_t* out_) ShapeInferContext_GetInputCount;

    /**
     * Get type and shape info of an input
     *
     * \param[in] context
     * \param[in] index The index of the input
     * \param[out] info Type shape info of the input
     *
     * \since Version 1.17.
     */
    OrtStatusPtr function(const(OrtShapeInferContext)* context, size_t index, OrtTensorTypeAndShapeInfo** info) ShapeInferContext_GetInputTypeShape;

    /**
     * Get attribute from OrtShapeInferContext. Note that OrtShapeInferContext is a per-node context, one could only read attribute from current node.
     *
     * \param[in] context
     * \param[in] attr_name Name of the attribute
     * \param[out] attr Handle of the attribute fetched
     *
     * \since Version 1.17.
     */
    OrtStatusPtr function(const(OrtShapeInferContext)* context, const(char)* attr_name, const(
            OrtOpAttr*)* attr) ShapeInferContext_GetAttribute;

    /**
     * Set type and shape info of an ouput
     *
     * \param[in] context
     * \param[in] index The index of the ouput
     * \param[out] info Type shape info of the output
     *
     * \since Version 1.17.
     */
    OrtStatusPtr function(const(OrtShapeInferContext)* context, size_t index, const(
            OrtTensorTypeAndShapeInfo)* info) ShapeInferContext_SetOutputTypeShape;

    /**
     * Set symbolic shape to type shape info
     *
     * \param[in] info Type shape info
     * \param[in] dim_params Symbolic strings
     * \param[in] dim_params_length Number of strings
     *
     * \since Version 1.17.
     */
    OrtStatusPtr function(OrtTensorTypeAndShapeInfo* info, const(char)*[] dim_params, size_t dim_params_length) SetSymbolicDimensions;

    /**
     * Read contents of an attribute to data
     *
     * \param[in] op_attr
     * \param[in] type Attribute type
     * \param[out] data Memory address to save raw content of the attribute
     * \param[in] len Number of bytes allowed to store in data
     * \param[out] out Number of bytes required to save the data when the call failed, or the real number of bytes saved to data on success
     *
     * \since Version 1.17.
     */
    OrtStatusPtr function(const(OrtOpAttr)* op_attr, OrtOpAttrType type, void* data, size_t len, size_t* out_) ReadOpAttr;

    /** \brief Set whether to use deterministic compute.
     *
     * Default is false. If set to true, this will enable deterministic compute for GPU kernels where possible.
     * Note that this most likely will have a performance cost.
     *
     * \param[in] options
     * \param[in] value
     *
     * \since Version 1.17.
     */
    OrtStatusPtr function(OrtSessionOptions* options, bool value) SetDeterministicCompute;

    /**
     * Run fn in parallel
     *
     * \param[in] context
     * \param[in] fn Function accepting usr_data and an integer as iterator
     * \param[in] total The number of times fn is to be invoked
     * \param[in] num_batch Number of batches by which the "total" is to be divided in maximum. When zero, there is no limit
     * \param[in] usr_data User data to be passed back to fn
     *
     * \since Version 1.17.
     */
    OrtStatusPtr function(const(OrtKernelContext)* context, void function(void*, size_t) fn, size_t total, size_t num_batch, void* usr_data) KernelContext_ParallelFor;

    /** \brief Append OpenVINO execution provider to the session options
     *
     * If OpenVINO is not available (due to a non OpenVINO enabled build, or if OpenVINO is not installed on the system), this function will fail.
     *
     * \param[in] options
     * \param[in] provider_options_keys
     * \param[in] provider_options_values
     * \param[in] num_keys
     *
     * \snippet{doc} snippets.dox OrtStatus Return Value
     */
    OrtStatusPtr function(
        OrtSessionOptions* options,
        const(char*)* provider_options_keys,
        const(char*)* provider_options_values,
        size_t num_keys) SessionOptionsAppendExecutionProvider_OpenVINO_V2;
}

/*
 * Steps to use a custom op:
 *   1 Create an OrtCustomOpDomain with the domain name used by the custom ops
 *   2 Create an OrtCustomOp structure for each op and add them to the domain
 *   3 Call OrtAddCustomOpDomain to add the custom domain of ops to the session options
 */

// Specifies some characteristics of inputs/outputs of custom ops:
// Specify if the inputs/outputs are one of:
// 1) Non-optional (input/output must be present in the node)
// 2) Optional (input/output may be absent in the node)
// 3) Variadic: A variadic input or output specifies N (i.e., the minimum arity) or more operands.
//              Only the last input or output of a custom op may be marked as variadic.
//              The homogeneity of the variadic input or output determines whether all operands must be of the same
//              tensor element type.
enum OrtCustomOpInputOutputCharacteristic {
    INPUT_OUTPUT_REQUIRED = 0,
    INPUT_OUTPUT_OPTIONAL = 1,
    INPUT_OUTPUT_VARIADIC = 2
}

alias INPUT_OUTPUT_REQUIRED = OrtCustomOpInputOutputCharacteristic.INPUT_OUTPUT_REQUIRED;
alias INPUT_OUTPUT_OPTIONAL = OrtCustomOpInputOutputCharacteristic.INPUT_OUTPUT_OPTIONAL;
alias INPUT_OUTPUT_VARIADIC = OrtCustomOpInputOutputCharacteristic.INPUT_OUTPUT_VARIADIC;

/*
 * The OrtCustomOp structure defines a custom op's schema and its kernel callbacks. The callbacks are filled in by
 * the implementor of the custom op.
 */
struct OrtCustomOp {
    uint version_; // Must be initialized to ORT_API_VERSION

    // This callback creates the kernel, which is a user defined
    // parameter that is passed to the Kernel* callbacks below. It is
    // recommended to use CreateKernelV2 which allows for a safe error
    // propagation by returning an OrtStatusPtr.
    void* function(
        const(OrtCustomOp)* op,
        const(OrtApi)* api,
        const(OrtKernelInfo)* info) CreateKernel;

    // Returns the name of the op
    const(char)* function(const(OrtCustomOp)* op) GetName;

    // Returns the type of the execution provider, return nullptr to use CPU execution provider
    const(char)* function(const(OrtCustomOp)* op) GetExecutionProviderType;

    // Returns the count and types of the input & output tensors
    ONNXTensorElementDataType function(const(OrtCustomOp)* op, size_t index) GetInputType;
    size_t function(const(OrtCustomOp)* op) GetInputTypeCount;
    ONNXTensorElementDataType function(const(OrtCustomOp)* op, size_t index) GetOutputType;
    size_t function(const(OrtCustomOp)* op) GetOutputTypeCount;

    // Perform a computation step.  It is recommended to use
    // KernelComputeV2 which allows for a safe error propagation by
    // returning an OrtStatusPtr.
    void function(void* op_kernel, OrtKernelContext* context) KernelCompute;
    void function(void* op_kernel) KernelDestroy;

    // Returns the characteristics of the input & output tensors
    OrtCustomOpInputOutputCharacteristic function(const(OrtCustomOp)* op, size_t index) GetInputCharacteristic;
    OrtCustomOpInputOutputCharacteristic function(const(OrtCustomOp)* op, size_t index) GetOutputCharacteristic;

    // Returns the memory type of the input tensors. This API allows the custom op
    // to place the inputs on specific devices. By default, it returns
    // OrtMemTypeDefault, which means the input is placed on the default device for
    // the execution provider. If the inputs need to be with different memory tyeps,
    // this function can be overridden to return the specific memory types.
    OrtMemType function(const(OrtCustomOp)* op, size_t index) GetInputMemoryType;

    // Returns the minimum number of input arguments expected for the variadic input.
    // Applicable only for custom ops that have a variadic input.
    int function(const(OrtCustomOp)* op) GetVariadicInputMinArity;

    // Returns true (non-zero) if all arguments of a variadic input have to be of the same type (homogeneous),
    // and false (zero) otherwise.
    // Applicable only for custom ops that have a variadic input.
    int function(const(OrtCustomOp)* op) GetVariadicInputHomogeneity;

    // Returns the minimum number of output values expected for the variadic output.
    // Applicable only for custom ops that have a variadic output.
    int function(const(OrtCustomOp)* op) GetVariadicOutputMinArity;

    // Returns true (non-zero) if all outputs values of a variadic output have to be of the same type (homogeneous),
    // and false (zero) otherwise.
    // Applicable only for custom ops that have a variadic output.
    int function(const(OrtCustomOp)* op) GetVariadicOutputHomogeneity;

    // Create the kernel state which is passed to each compute call.
    OrtStatusPtr function(
        const(OrtCustomOp)* op,
        const(OrtApi)* api,
        const(OrtKernelInfo)* info,
        void** kernel) CreateKernelV2;

    // Perform the computation step.
    OrtStatusPtr function(void* op_kernel, OrtKernelContext* context) KernelComputeV2;

    OrtStatusPtr function(const(OrtCustomOp)* op, OrtShapeInferContext*) InferOutputShapeFn;

    // Get start range
    int function(const(OrtCustomOp)* op) GetStartVersion;
    int function(const(OrtCustomOp)* op) GetEndVersion;
}

/*
 * This is the old way to add the CUDA provider to the session, please use SessionOptionsAppendExecutionProvider_CUDA above to access the latest functionality
 * This function always exists, but will only succeed if Onnxruntime was built with CUDA support and the CUDA provider shared library exists
 *
 * \param device_id CUDA device id, starts from zero.
 */
OrtStatusPtr OrtSessionOptionsAppendExecutionProvider_CUDA(OrtSessionOptions* options, int device_id);

/*
 * This is the old way to add the ROCm provider to the session, please use
 * SessionOptionsAppendExecutionProvider_ROCM above to access the latest functionality
 * This function always exists, but will only succeed if Onnxruntime was built with
 * HIP support and the ROCm provider shared library exists
 *
 * \param device_id HIP device id, starts from zero.
 */
OrtStatusPtr OrtSessionOptionsAppendExecutionProvider_ROCM(OrtSessionOptions* options, int device_id);

/*
 * This is the old way to add the MIGraphX provider to the session, please use
 * SessionOptionsAppendExecutionProvider_MIGraphX above to access the latest functionality
 * This function always exists, but will only succeed if Onnxruntime was built with
 * HIP support and the MIGraphX provider shared library exists
 *
 * \param device_id HIP device id, starts from zero.
 */
OrtStatusPtr OrtSessionOptionsAppendExecutionProvider_MIGraphX(
    OrtSessionOptions* options, int device_id);

/*
 * This is the old way to add the oneDNN provider to the session, please use
 * SessionOptionsAppendExecutionProvider_oneDNN above to access the latest functionality
 * This function always exists, but will only succeed if Onnxruntime was built with
 * oneDNN support and the oneDNN provider shared library exists
 *
 * \param use_arena zero: false. non-zero: true.
 */
OrtStatusPtr OrtSessionOptionsAppendExecutionProvider_Dnnl(OrtSessionOptions* options, int use_arena);

/*
 * This is the old way to add the TensorRT provider to the session, please use SessionOptionsAppendExecutionProvider_TensorRT_V2 above to access the latest functionality
 * This function always exists, but will only succeed if Onnxruntime was built with TensorRT support and the TensorRT provider shared library exists
 *
 * \param device_id CUDA device id, starts from zero.
 */
OrtStatusPtr OrtSessionOptionsAppendExecutionProvider_Tensorrt(
    OrtSessionOptions* options, int device_id);

/// @}
