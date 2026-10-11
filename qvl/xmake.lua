includes("xmake/rules/*.lua")

add_requires("lighter 0.3.0")
-- Built by xmake rather than found on the host: a system library lives
-- outside the Pixi sysroot the compiler searches.
add_requires("pcre2", {system = false})
add_requires("utf8proc", {system = false})
add_requires("stb", {system = false})

-- OpenVINO comes from the Pixi environment (conda-forge libopenvino). Only the
-- runtime library is linked; libopenvino-dev also carries every model frontend.
package("openvino")
    on_fetch(function (package)
        local prefix = os.getenv("CONDA_PREFIX")
        if prefix then
            return {includedirs = path.join(prefix, "include"), linkdirs = path.join(prefix, "lib"), links = "openvino"}
        end
    end)
package_end()
add_requires("openvino")

target("qvl-server")
    set_kind("binary")
    add_rules("utils.version.cpp")
    add_files("main.cpp")
    add_packages("lighter", "openvino")
