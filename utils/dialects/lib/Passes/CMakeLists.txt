add_mlir_conversion_library(MLIRPolarPasses
        RelationDestructorPass.cpp
        DeployDefaultFunctions.cpp
        BarrierLegalizerPass.cpp
        LegalizeRTForLoweringPass.cpp
        ReleaseDependencyPass.cpp
        RuntimeShapeInferencePass.cpp
        KernelMaterializerPass.cpp
        KernelOutliningPass.cpp
        SaveKernelPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_BINARY_DIR}/include

        DEPENDS
        MLIRPolarGraph
        )

target_link_libraries(MLIRPolarPasses
        PUBLIC
        MLIRIR
        MLIRPass
        MLIRPolarGraph
        MLIRTransforms
        MLIRCompute
        )
