-- -- xmake.lua (适配平面目录结构)
-- set_project("OnnxCppInference")
-- set_version("1.0.0")
-- set_default_target("onnx_infer") -- 设置默认构建目标

-- -- 设置语言标准和警告级别
-- set_languages("cxx17") -- 或 "cxx20", "cxxlatest"
-- set_warnings("all")
-- set_optimize("release") -- 或 "debug" 用于调试

-- 定义可执行目标 'onnx_infer'
target("onnx_infer")
    set_kind("binary") -- 类型为可执行文件
    if is_plat("macosx") then
        add_rpathdirs("@loader_path") -- @loader_path 指向可执行文件所在的目录
        print("Added @loader_path to RPATH for macOS")
    end
    -- 添加源文件 (都在根目录)
    add_files("main.cpp")
    -- 不需要添加 infer.cpp 了

    -- 添加头文件搜索路径
    -- xmake 默认会搜索当前目录，所以 infer.hpp 能被 main.cpp 找到
    -- 添加 ONNX Runtime 的 include 目录
    add_includedirs("vendor/onnxruntime/include", {public = true})

    -- 添加库文件搜索路径
    add_linkdirs("vendor/onnxruntime/lib")

    -- 链接 ONNX Runtime 库
    -- 库名称通常是 onnxruntime (不带lib前缀和.so/.lib/.dylib后缀)
    -- xmake 会自动处理平台差异
    add_links("onnxruntime")

    -- [[ 根据平台和使用的 Execution Provider 添加特定依赖 ]]
    -- 例如 Linux 通常需要 pthread 和 dl
    -- if is_plat("linux") then
    --     add_syslinks("pthread", "dl")
    -- end
    -- 例如 macOS 使用 CoreML EP 可能需要链接框架
    -- if is_plat("macosx") then
    --     -- add_frameworks("CoreFoundation", "Foundation", "CoreML")
    -- end

    -- 打印消息，方便调试确认路径
    print("Using ONNX Runtime from: ./vendor/onnxruntime")

    set_languages("c++17")
    after_link(function (target)
        -- 只在构建 onnx_infer 目标且是 macOS 平台时执行
        if target:name() == "onnx_infer" and is_plat("macosx") then
            local sourcedir = path.join(os.projectdir(), "vendor/onnxruntime/lib")
            local targetdir = target:targetdir() -- 获取目标文件输出目录 (e.g., build/macosx/arm64/release)

            print("--- Running after_link script for target: " .. target:name() .. " ---")
            print("Source directory for dylib: " .. sourcedir)
            print("Target directory for copy: " .. targetdir)

            -- 确保目标目录存在
            if not os.isdir(targetdir) then
                print("Creating target directory: " .. targetdir)
                os.mkdirs(targetdir) -- Use recursive mkdir
            else
                 print("Target directory already exists.")
            end

            -- 查找 dylib 文件 (名字可能包含版本号)
            local dylib_pattern = "libonnxruntime.*.dylib"
            local dylib_full_source_path = nil -- 存储找到的 dylib 的完整源路径
            local dylib_filename = nil       -- 存储找到的 dylib 的文件名

            print("Searching for '" .. dylib_pattern .. "' in " .. sourcedir)
            local found_files = os.files(path.join(sourcedir, dylib_pattern)) -- os.files 返回的是完整路径列表

            if #found_files == 0 then
                 print("ERROR: No files found matching pattern '" .. dylib_pattern .. "'!")
            else
                for i, file_path_absolute in ipairs(found_files) do
                    print("Found file candidate [" .. i .. "]: " .. file_path_absolute)
                    -- 假设第一个匹配的就是我们要找的
                    if i == 1 then
                        -- *** 修正 ***
                        -- file_path_absolute 已经是完整的源路径了
                        dylib_full_source_path = file_path_absolute
                        -- 使用 path.filename() 提取纯粹的文件名
                        dylib_filename = path.filename(file_path_absolute)
                        print("Selected dylib full path: " .. dylib_full_source_path)
                        print("Selected dylib filename: " .. dylib_filename)
                        break -- 找到第一个就跳出循环
                    end
                end
            end

            -- 如果找到了 dylib 文件和文件名，则执行复制
            if dylib_full_source_path and dylib_filename then
                -- *** 修正 ***
                -- 构建目标路径：目标目录 + 文件名
                local destination_path = path.join(targetdir, dylib_filename)
                print("Attempting to copy '" .. dylib_full_source_path .. "' to '" .. destination_path .. "'")

                -- 执行复制 (源路径是 dylib_full_source_path)
                local copy_result, errmsg = os.cp(dylib_full_source_path, destination_path)
                if copy_result then
                    print("Copy successful!")
                else
                    print("ERROR: Failed to copy dylib!")
                    if errmsg then print("Reason from os.cp: " .. errmsg) end
                end
            else
                print("ERROR: Could not determine the dylib path or filename to copy.")
            end
            print("--- Finished after_link script ---")
        end
    end)


-- 设置默认运行的目标 (可选)
-- after_build(function (target)
--     if target:name() == "onnx_infer" and target:kind() == "binary" then
--         os.exec("echo Run with: ./build/.../onnx_infer preproc_encoder.onnx path/to/audio.wav")
--     end
-- end)