#include "prompt.h"

#include <algorithm>

#include <utf8proc.h>

namespace qvl::prompt {

namespace {

/// The instruction with a full stop unless its last character is punctuation
/// (Unicode category P*), as the reference embedder normalizes it.
std::string punctuated(std::string_view instruction) {
    while (!instruction.empty() && (instruction.front() == ' ' || instruction.front() == '\n' || instruction.front() == '\t')) {
        instruction.remove_prefix(1);
    }
    while (!instruction.empty() && (instruction.back() == ' ' || instruction.back() == '\n' || instruction.back() == '\t')) {
        instruction.remove_suffix(1);
    }
    std::string out(instruction);
    if (out.empty()) return out;

    // Step back to the start of the last UTF-8 sequence.
    usize last = out.size() - 1;
    while (last > 0 && (static_cast<unsigned char>(out[last]) & 0xC0) == 0x80) --last;
    utf8proc_int32_t codepoint = 0;
    utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t *>(out.data() + last), static_cast<utf8proc_ssize_t>(out.size() - last),
                     &codepoint);
    switch (utf8proc_category(codepoint)) {
        case UTF8PROC_CATEGORY_PC:
        case UTF8PROC_CATEGORY_PD:
        case UTF8PROC_CATEGORY_PS:
        case UTF8PROC_CATEGORY_PE:
        case UTF8PROC_CATEGORY_PI:
        case UTF8PROC_CATEGORY_PF:
        case UTF8PROC_CATEGORY_PO: break;
        default: out += '.';
    }
    return out;
}

void append_image(std::string &out, u32 tokens) {
    out += "<|vision_start|>";
    for (u32 i = 0; i < tokens; ++i) out += "<|image_pad|>";
    out += "<|vision_end|>";
}

std::string turns(std::string_view system, std::string_view user) {
    std::string out = "<|im_start|>system\n";
    out += system;
    out += "<|im_end|>\n<|im_start|>user\n";
    out += user;
    out += "<|im_end|>\n<|im_start|>assistant\n";
    return out;
}

/// Image then text after `prefix`, or "NULL" when the side is empty.
void append_side(std::string &out, std::string_view prefix, const Content &content) {
    out += prefix;
    if (content.image_tokens == 0 && !content.text) {
        out += "NULL";
        return;
    }
    if (content.image_tokens > 0) append_image(out, content.image_tokens);
    if (content.text) out += *content.text;
}

} // namespace

std::string embedding(std::string_view instruction, const Content &content) {
    std::string user;
    if (content.image_tokens == 0 && !content.text) {
        user = "NULL";
    } else {
        if (content.image_tokens > 0) append_image(user, content.image_tokens);
        if (content.text) user += *content.text;
    }
    return turns(punctuated(instruction.empty() ? k_embedding_instruction : instruction), user);
}

std::string rerank(std::string_view instruction, const Content &query, const Content &document) {
    std::string user = "<Instruct>: ";
    user += instruction.empty() ? k_rerank_instruction : instruction;
    append_side(user, "<Query>:", query);
    append_side(user, "\n<Document>:", document);
    return turns(k_rerank_system, user);
}

std::expected<std::vector<i64>, std::string> position_ids(std::span<const i32> ids, i32 image_token, std::span<const vision::Grid> grids) {
    const usize n = ids.size();
    std::vector<i64> out(3 * n);
    auto set = [&](usize at, i64 t, i64 h, i64 w) {
        out[at] = t;
        out[n + at] = h;
        out[2 * n + at] = w;
    };

    i64 next = 0;
    usize image = 0;
    for (usize i = 0; i < n;) {
        if (ids[i] != image_token) {
            set(i, next, next, next);
            ++next;
            ++i;
            continue;
        }
        if (image >= grids.size()) return std::unexpected("more images in the prompt than grids");
        const auto grid = grids[image++];
        const u32 rows = grid.h / 2;
        const u32 columns = grid.w / 2;
        const usize run = static_cast<usize>(rows) * columns;
        if (i + run > n || !std::ranges::all_of(ids.subspan(i, run), [&](i32 id) { return id == image_token; }) ||
            (i + run < n && ids[i + run] == image_token)) {
            return std::unexpected("an image's placeholders do not match its grid");
        }
        for (usize k = 0; k < run; ++k) {
            set(i + k, next, next + static_cast<i64>(k / columns), next + static_cast<i64>(k % columns));
        }
        next += std::max(rows, columns);
        i += run;
    }
    if (image != grids.size()) return std::unexpected("fewer images in the prompt than grids");
    return out;
}

} // namespace qvl::prompt
