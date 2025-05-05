
-- 定义项目名称和默认目标类型
-- set_project("OnnxCppInference")
-- set_version("1.0.0")
-- set_default_target("onnx_infer")


-- 添加可执行目标
target("onnx_infer")
    set_kind("binary")
    -- 添加源文件 (假设放在 src 目录下)
    add_files("src/main.cpp")
    --[[
        如果你将 InferenceEngine 的实现放在 infer.cpp 中，
        而不是纯头文件，也需要添加它：
        add_files("src/infer.cpp")
    --]]

    -- 添加头文件搜索路径
    -- 假设 download_dep.py 将库解压到 ./vendor/onnxruntime
    add_includedirs("vendor/onnxruntime/include", {public = true})
    -- 添加 infer.hpp 所在的目录 (如果 main.cpp 需要 include 它)
    add_includedirs("src", {public = true})

    -- 添加库搜索路径
    add_linkdirs("vendor/onnxruntime/lib")

    -- 链接 ONNX Runtime 库
    -- 库文件名可能因平台而异 (onnxruntime.lib, libonnxruntime.so, libonnxruntime.dylib)
    -- xmake 通常能自动处理，或者你可能需要条件判断
    add_links("onnxruntime")

    -- [[ 可选：如果使用 Execution Providers (如 CUDA, TensorRT), 可能需要链接额外的库或添加特定的编译/链接标志 ]]
    -- if is_plat("windows") then
    --     -- Windows specific settings
    -- elseif is_plat("linux") then
    --     add_syslinks("pthread", "dl") -- Common Linux dependencies for ORT
    -- elseif is_plat("macosx") then
    --     -- macOS specific settings, e.g., linking CoreFoundation, CoreML frameworks
    --     -- add_frameworks("CoreFoundation", "CoreML", "Foundation")
    -- end

    -- [[ 可选：如果你使用音频库 (如 libsndfile) 加载音频 ]]
    -- add_requires("libsndfile") -- 如果通过 xmake-repo 管理
    -- add_packages("libsndfile")
    -- 或者手动配置 include/lib/link

    -- 打印消息，确认库路径被使用
    print("Using ONNX Runtime from: ./vendor/onnxruntime")