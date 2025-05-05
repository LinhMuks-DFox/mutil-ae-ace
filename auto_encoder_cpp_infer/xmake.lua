-- xmake.lua (基于你的版本，添加 Linux 支持)

-- [[ 你注释掉的全局设置 ]]
-- set_project("OnnxCppInference")
-- set_version("1.0.0")
-- set_default_target("onnx_infer")
-- set_languages("cxx17")
-- set_warnings("all")
-- set_optimize("release")
-- add_requires("portaudio", { system = false }) -- 尝试查找或构建本地版本

-- 定义可执行目标 'onnx_infer'
target("onnx_infer")
    set_kind("binary") -- 类型为可执行文件

    -- --- 添加 RPATH 设置 (区分平台) ---
    if is_plat("macosx") then
        add_rpathdirs("@loader_path") -- @loader_path 指向可执行文件所在的目录
        print("Added @loader_path to RPATH for macOS")
    elseif is_plat("linux") then
        -- Linux: 使用 $ORIGIN (指向可执行文件自身所在的目录)
        add_rpathdirs("$ORIGIN")
        print("Added $ORIGIN to RPATH for Linux")
    end
    -- ------------------------------------

    -- 添加源文件 (都在根目录)
    add_files("main.cpp")
    -- 不需要添加 infer.cpp 了

    -- 添加头文件搜索路径
    add_includedirs("vendor/onnxruntime/include", {public = true})

    -- 添加库文件搜索路径
    add_linkdirs("vendor/onnxruntime/lib")

    -- 链接 ONNX Runtime 库
    add_links("onnxruntime")
    add_packages("portaudio")

    -- [[ 根据平台和使用的 Execution Provider 添加特定依赖 ]]
    if is_plat("linux") then
        -- Linux 下链接 ONNX Runtime 可能需要这些系统库
         add_syslinks("pthread", "dl")
         print("Added syslinks pthread, dl for Linux")
    end
    -- if is_plat("macosx") then
    --    -- add_frameworks("CoreFoundation", "Foundation", "CoreML")
    -- end

    print("Using ONNX Runtime from: ./vendor/onnxruntime")

    -- 确保 C++ 标准设置在 target 内部或外部均可，这里放在内部
    set_languages("c++17")

    -- --- 添加链接后脚本来复制共享库 (区分平台) ---
    after_link(function (target)
        -- 只为 onnx_infer 可执行目标运行
        if target:name() == "onnx_infer" and target:kind() == "binary" then

            -- 定义一个内部辅助函数用于查找和复制
            local function copy_shared_lib(sourcedir, targetdir, pattern)
                print("Searching for pattern '" .. pattern .. "' in " .. sourcedir)
                local lib_full_source_path = nil
                local lib_filename = nil
                -- 使用 os.files 获取文件列表 (之前确认过可以获取绝对路径)
                local found_files = os.files(path.join(sourcedir, pattern))

                if #found_files == 0 then
                    print("Warning: No files found matching pattern '" .. pattern .. "'!")
                    return -- 找不到就返回
                end

                -- 通常取列表中的第一个
                lib_full_source_path = found_files[1]
                lib_filename = path.filename(lib_full_source_path)
                print("Found library candidate: " .. lib_full_source_path)
                print("Filename: " .. lib_filename)

                local destination_path = path.join(targetdir, lib_filename)
                print("Attempting to copy '" .. lib_full_source_path .. "' to '" .. destination_path .. "'")

                local copy_result, errmsg = os.cp(lib_full_source_path, destination_path)
                if copy_result then
                    print("Copy successful for: " .. lib_filename)
                else
                    print("ERROR: Failed to copy shared library: " .. lib_filename)
                    if errmsg then print("Reason from os.cp: " .. errmsg) end
                end
            end -- end function copy_shared_lib


            local sourcedir = path.join(os.projectdir(), "vendor/onnxruntime/lib")
            local targetdir = target:targetdir() -- 获取目标输出目录

            print("--- Running after_link script for target: " .. target:name() .. " ---")
            print("Source directory for shared libs: " .. sourcedir)
            print("Target directory for copy: " .. targetdir)

            -- 确保目标目录存在
            if not os.isdir(targetdir) then
                print("Creating target directory: " .. targetdir)
                os.mkdirs(targetdir)
            else
                 print("Target directory already exists.")
            end

            -- --- 根据平台执行复制 ---
            if is_plat("macosx") then
                print("Platform: macOS. Copying dylib...")
                copy_shared_lib(sourcedir, targetdir, "libonnxruntime.*.dylib")
            elseif is_plat("linux") then
                print("Platform: Linux. Copying .so files...")
                -- 复制版本化的 .so 文件 (如 libonnxruntime.so.1.17.3)
                copy_shared_lib(sourcedir, targetdir, "libonnxruntime.so.*")
                -- 复制 .so 符号链接 (或文件本身，如 libonnxruntime.so)
                copy_shared_lib(sourcedir, targetdir, "libonnxruntime.so")
            elseif is_plat("windows") then
                print("Platform: Windows. Copying dll...")
                 copy_shared_lib(sourcedir, targetdir, "onnxruntime.dll")
                 -- copy_shared_lib(sourcedir, targetdir, "onnxruntime.lib")
            else
                 print("Warning: Unknown platform for after_link copy step.")
            end
            -- -----------------------

            print("--- Finished after_link script ---")
        end -- end if target name matches
    end) -- end after_link
    -- -------------------------------------------------