find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    CUDA_ADD_LIBRARY(gpu STATIC
        api.cu
        )
    
    target_include_directories(gpu PRIVATE ../)
endif()