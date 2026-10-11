set_default(false)

-- Golden data from tools/make_fixtures.py.
add_defines("QVL_FIXTURE_DIR=\"" .. path.join(os.scriptdir(), "fixtures"):gsub("\\", "/") .. "\"")

includes("*/xmake.lua")
