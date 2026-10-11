#include <cstdio>
#include <exception>
#include <string>
#include <string_view>

#include <openvino/openvino.hpp>

#include <xmake/version/qvl.hpp>

int main(int argc, char **argv) {
    const std::string_view command = argc > 1 ? argv[1] : "";
    if (command == "--version") {
        std::printf("qvl-server %.*s (%.*s%s)\n", static_cast<int>(xmake::version::version.size()), xmake::version::version.data(),
                    static_cast<int>(xmake::version::commit.size()), xmake::version::commit.data(), xmake::version::dirty ? "-dirty" : "");
        return 0;
    }
    if (command == "--devices") {
        try {
            ov::Core core;
            std::printf("OpenVINO %s\n", ov::get_openvino_version().buildNumber);
            for (const auto &device : core.get_available_devices()) {
                std::printf("%s: %s\n", device.c_str(), core.get_property(device, ov::device::full_name).c_str());
            }
        } catch (const std::exception &error) {
            std::fprintf(stderr, "qvl-server: %s\n", error.what());
            return 1;
        }
        return 0;
    }
    std::fprintf(stderr, "usage: qvl-server --version | --devices\n");
    return 2;
}
