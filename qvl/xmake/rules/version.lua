-- Generate C++ product version and source provenance metadata.
-- The generated header is available as <xmake/version/qvl.hpp>.
local rule_name = "utils.version.cpp"

local function _header_content(version, commit, dirty)
    return string.format([=[#pragma once

#include <string_view>

namespace xmake::version {

inline constexpr std::string_view version{"%s"};
inline constexpr std::string_view commit{"%s"};
inline constexpr bool dirty{%s};

} // namespace xmake::version
]=], version, commit, dirty and "true" or "false")
end

local function _generate_header(target)
    import("lib.detect.find_tool")

    local git = find_tool("git")
    local function git_output(argv)
        if not git then
            return nil
        end
        return try {function()
            return os.iorunv(git.program, argv, {curdir = os.projectdir()}):trim()
        end}
    end

    local version = target:version()
    if not version or #version == 0 then
        raise("%s: the project version is unavailable", rule_name)
    end

    local commit = os.getenv("QVL_SOURCE_COMMIT") or git_output({"rev-parse", "HEAD"}) or ""
    local dirty_override = os.getenv("QVL_SOURCE_DIRTY")
    local dirty
    if dirty_override then
        if dirty_override ~= "0" and dirty_override ~= "1" then
            raise("%s: QVL_SOURCE_DIRTY must be 0 or 1", rule_name)
        end
        dirty = dirty_override == "1"
    else
        local status = commit ~= "" and git_output({"status", "--porcelain=v1", "--untracked-files=normal"}) or ""
        dirty = status ~= nil and status ~= ""
    end
    local headerroot = path.join(target:autogendir(), "rules", rule_name)
    local headerdir = path.join(headerroot, "xmake", "version")
    local headerfile = path.join(headerdir, "qvl.hpp")
    local content = _header_content(version, commit, dirty)

    if not os.isfile(headerfile) or io.readfile(headerfile) ~= content then
        os.mkdir(headerdir)
        io.writefile(headerfile, content)
    end

    target:add("includedirs", headerroot, {public = true})
end

rule(rule_name)
    on_load(_generate_header)
