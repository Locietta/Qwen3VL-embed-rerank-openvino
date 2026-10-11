set_xmakever("3.0.9")
set_project("qvl")

local project_version = os.getenv("PIXI_PROJECT_VERSION")
if not project_version then
    raise("this project reads its version from Pixi: use `pixi run xmake ...` or develop inside `pixi shell`")
end
set_version(project_version)

if not is_host("linux") then
    raise("qvl builds on Linux only; from Windows use `pixi run test-remote-linux --host <host>`")
end

-- use releasedbg for development builds, ship with release
add_rules("mode.release", "mode.releasedbg")

set_languages("cxxlatest")

option("sanitizers")
    set_default(false)
    set_showmenu(true)
option_end()

if has_config("sanitizers") then
    add_cxxflags("-fsanitize=address,undefined", "-fno-omit-frame-pointer", {tools = {"gcc", "gxx"}, force = true})
    add_ldflags("-fsanitize=address,undefined", {tools = {"gcc", "gxx"}, force = true})
end

local contract_semantic = "enforce"
if is_mode("release") then
    contract_semantic = "ignore"
end

-- Enable reflection and contracts on gcc 16+
add_cxxflags(
    "-freflection",
    "-fcontracts",
    "-fcontract-evaluation-semantic=" .. contract_semantic,
    {tools = {"gcc", "gxx"}, force = true})

-- The toolchain, OpenVINO and the C dependencies all come from the Pixi
-- environment, so every xmake invocation has to happen under `pixi run`. The
-- script scope has no error primitives, hence the option below rather than a
-- plain check here.
local conda_prefix = os.getenv("CONDA_PREFIX") or ""

option("__pixi_environment")
    set_showmenu(false)
    on_check(function (option)
        if not os.getenv("CONDA_PREFIX") then
            raise("this project builds inside its Pixi environment: use `pixi run build`, or `pixi run xmake ...`")
        end
        option:enable(true)
    end)
option_end()

-- The shared libraries resolved out of the Pixi environment are not on the
-- system loader path, so record it in the binary the way conda's own
-- activation scripts do.
add_rpathdirs(path.join(conda_prefix, "lib"))

add_repositories("loia-pinned xmake", {rootdir = os.scriptdir()})
add_moduledirs("xmake/modules")

option("__pixi_package_manager")
    set_showmenu(false)
    on_check(function (option)
        import("package.manager.pixi.register")()
        option:enable(true)
    end)
option_end()

includes("xmake/rules/*.lua")
add_rules("ci.test-output")

add_includedirs(".")

includes("*/xmake.lua")
