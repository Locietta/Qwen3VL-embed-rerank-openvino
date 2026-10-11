#include "tokenizer.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <variant>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <glaze/json.hpp>
#include <pcre2.h>
#include <utf8proc.h>

namespace qvl::tokenizer {

namespace {

struct AddedTokenEntry {
    i32 id = 0;
    std::string content;
};

struct ModelEntry {
    std::string type;
    std::unordered_map<std::string, i32> vocab;
    /// "left right" (older files) or ["left", "right"].
    std::vector<std::variant<std::string, std::vector<std::string>>> merges;
    bool ignore_merges = false;
};

struct FileEntry {
    std::vector<AddedTokenEntry> added_tokens;
    glz::generic normalizer;
    glz::generic pre_tokenizer;
    glz::generic post_processor;
    ModelEntry model;
};

const glz::generic *field(const glz::generic &object, std::string_view key) {
    if (!object.is_object()) return nullptr;
    const auto &map = object.get_object();
    const auto it = map.find(key);
    return it == map.end() ? nullptr : &it->second;
}

std::string_view string_field(const glz::generic &object, std::string_view key) {
    const auto *value = field(object, key);
    return value && value->is_string() ? std::string_view(value->get_string()) : std::string_view();
}

std::string utf8(u32 codepoint) {
    std::string out;
    if (codepoint < 0x80) {
        out += static_cast<char>(codepoint);
    } else if (codepoint < 0x800) {
        out += static_cast<char>(0xC0 | (codepoint >> 6));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else {
        out += static_cast<char>(0xE0 | (codepoint >> 12));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    return out;
}

/// GPT-2's bytes_to_unicode(): printable bytes map to themselves, the rest to
/// U+0100 onwards, so every byte is one visible character.
std::array<std::string, 256> byte_symbols() {
    std::array<bool, 256> printable{};
    for (u32 b = '!'; b <= '~'; ++b) printable[b] = true;
    for (u32 b = 0xA1; b <= 0xAC; ++b) printable[b] = true;
    for (u32 b = 0xAE; b <= 0xFF; ++b) printable[b] = true;
    std::array<std::string, 256> out;
    u32 next = 256;
    for (u32 b = 0; b < 256; ++b) {
        out[b] = utf8(printable[b] ? b : next++);
    }
    return out;
}

/// Byte length of the UTF-8 sequence starting with `lead` (1 for stray bytes).
usize sequence_length(unsigned char lead) {
    if (lead >= 0xF0) return 4;
    if (lead >= 0xE0) return 3;
    if (lead >= 0xC0) return 2;
    return 1;
}

} // namespace

struct Tokenizer::Self {
    std::unordered_map<std::string, i32> vocab;
    std::unordered_map<std::string, i32> ranks;
    bool ignore_merges = false;
    /// Added tokens, longest first, so a prefix never shadows a longer token.
    std::vector<AddedTokenEntry> added;
    std::unordered_map<std::string, i32> added_ids;
    bool nfc = false;
    pcre2_code *split = nullptr;
    std::vector<i32> prefix;
    std::vector<i32> suffix;
    std::array<std::string, 256> bytes = byte_symbols();

    ~Self() {
        if (split) pcre2_code_free(split);
    }

    void encode_word(std::string_view word, std::vector<i32> &out) const {
        std::string mapped;
        for (unsigned char byte : word) mapped += bytes[byte];
        if (ignore_merges) {
            if (const auto it = vocab.find(mapped); it != vocab.end()) {
                out.push_back(it->second);
                return;
            }
        }

        // Each byte symbol is a one-character UTF-8 string, linked to its
        // neighbours; merging keeps the left symbol, so symbol 0 stays first.
        struct Symbol {
            std::string text;
            i32 prev = -1;
            i32 next = -1;
            bool alive = true;
        };
        std::vector<Symbol> symbols;
        for (usize i = 0; i < mapped.size();) {
            const auto length = sequence_length(static_cast<unsigned char>(mapped[i]));
            const auto index = static_cast<i32>(symbols.size());
            symbols.push_back({.text = mapped.substr(i, length), .prev = index - 1, .next = -1});
            if (index > 0) symbols[index - 1].next = index;
            i += length;
        }

        // The lowest-ranked pair merges first, the leftmost on ties, as in
        // `tokenizers`' BPE; a heap keeps this O(n log n) for long words.
        struct Candidate {
            i32 rank;
            i32 position;
            bool operator>(const Candidate &other) const { return rank != other.rank ? rank > other.rank : position > other.position; }
        };
        std::string key;
        auto rank_of = [&](i32 left) -> i32 {
            const auto right = symbols[left].next;
            if (right < 0) return -1;
            key.assign(symbols[left].text).append(" ").append(symbols[right].text);
            const auto it = ranks.find(key);
            return it == ranks.end() ? -1 : it->second;
        };
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;
        auto offer = [&](i32 left) {
            if (left < 0) return;
            if (const auto rank = rank_of(left); rank >= 0) queue.push({rank, left});
        };
        for (i32 i = 0; i + 1 < static_cast<i32>(symbols.size()); ++i) offer(i);
        while (!queue.empty()) {
            const auto top = queue.top();
            queue.pop();
            // Skip entries made stale by an earlier merge.
            if (!symbols[top.position].alive || rank_of(top.position) != top.rank) continue;
            auto &left = symbols[top.position];
            auto &right = symbols[left.next];
            left.text += right.text;
            right.alive = false;
            left.next = right.next;
            if (left.next >= 0) symbols[left.next].prev = top.position;
            offer(left.prev);
            offer(top.position);
        }
        for (i32 i = symbols.empty() ? -1 : 0; i >= 0; i = symbols[i].next) {
            if (const auto it = vocab.find(symbols[i].text); it != vocab.end()) out.push_back(it->second);
        }
    }

    /// A stretch of text without added tokens: normalize, split, BPE.
    void encode_segment(std::string_view text, std::vector<i32> &out) const {
        if (text.empty()) return;
        std::string normalized;
        if (nfc) {
            utf8proc_uint8_t *buffer = nullptr;
            const auto length =
                utf8proc_map(reinterpret_cast<const utf8proc_uint8_t *>(text.data()), static_cast<utf8proc_ssize_t>(text.size()), &buffer,
                             static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
            if (length >= 0) {
                normalized.assign(reinterpret_cast<const char *>(buffer), static_cast<usize>(length));
                text = normalized;
            }
            std::free(buffer);
        }

        auto *match = pcre2_match_data_create_from_pattern(split, nullptr);
        const auto *subject = reinterpret_cast<PCRE2_SPTR>(text.data());
        usize offset = 0;
        while (offset < text.size()) {
            const int found = pcre2_match(split, subject, text.size(), offset, 0, match, nullptr);
            if (found < 0) {
                encode_word(text.substr(offset), out);
                break;
            }
            const auto *ovector = pcre2_get_ovector_pointer(match);
            const usize start = ovector[0];
            const usize end = ovector[1];
            if (start > offset) encode_word(text.substr(offset, start - offset), out);
            if (end == start) {
                // An empty match cannot advance; take one character instead.
                const usize step = std::min(sequence_length(static_cast<unsigned char>(text[start])), text.size() - start);
                encode_word(text.substr(start, step), out);
                offset = start + step;
                continue;
            }
            encode_word(text.substr(start, end - start), out);
            offset = end;
        }
        pcre2_match_data_free(match);
    }

    std::vector<i32> encode(std::string_view text) const {
        std::vector<i32> out;
        usize begin = 0;
        for (usize i = 0; i < text.size();) {
            const AddedTokenEntry *hit = nullptr;
            for (const auto &token : added) {
                if (text.substr(i).starts_with(token.content)) {
                    hit = &token;
                    break;
                }
            }
            if (!hit) {
                ++i;
                continue;
            }
            encode_segment(text.substr(begin, i - begin), out);
            out.push_back(hit->id);
            i += hit->content.size();
            begin = i;
        }
        encode_segment(text.substr(begin), out);
        return out;
    }
};

Tokenizer::Tokenizer() noexcept = default;
Tokenizer::Tokenizer(Tokenizer &&) noexcept = default;
Tokenizer &Tokenizer::operator=(Tokenizer &&) noexcept = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(std::unique_ptr<Self> self) noexcept : self(std::move(self)) {}

std::expected<Tokenizer, std::string> Tokenizer::load(const std::filesystem::path &tokenizer_json) {
    std::ifstream stream(tokenizer_json, std::ios::binary);
    if (!stream) return std::unexpected("cannot read " + tokenizer_json.string());
    std::stringstream buffer;
    buffer << stream.rdbuf();
    const auto text = std::move(buffer).str();

    FileEntry file;
    if (auto ec = glz::read<glz::opts{.null_terminated = false, .error_on_unknown_keys = false}>(file, text)) {
        return std::unexpected("invalid tokenizer.json: " + glz::format_error(ec, text));
    }
    if (file.model.type != "BPE") return std::unexpected("unsupported tokenizer model: " + file.model.type);

    auto self = std::make_unique<Self>();
    self->vocab = std::move(file.model.vocab);
    self->ignore_merges = file.model.ignore_merges;
    for (usize rank = 0; rank < file.model.merges.size(); ++rank) {
        const auto &merge = file.model.merges[rank];
        std::string key;
        if (const auto *joined = std::get_if<std::string>(&merge)) {
            key = *joined;
        } else {
            const auto &pair = std::get<std::vector<std::string>>(merge);
            if (pair.size() != 2) return std::unexpected("malformed merge at rank " + std::to_string(rank));
            key = pair[0] + " " + pair[1];
        }
        self->ranks.try_emplace(std::move(key), static_cast<i32>(rank));
    }

    for (auto &token : file.added_tokens) {
        self->added_ids[token.content] = token.id;
        self->added.push_back(std::move(token));
    }
    std::ranges::stable_sort(self->added, [](const auto &a, const auto &b) { return a.content.size() > b.content.size(); });

    const auto normalizer = string_field(file.normalizer, "type");
    if (normalizer == "NFC") {
        self->nfc = true;
    } else if (!file.normalizer.is_null() && !normalizer.empty()) {
        return std::unexpected("unsupported normalizer: " + std::string(normalizer));
    }

    // The Split step of the pre-tokenizer carries the regex; the ByteLevel
    // step after it must not split again (use_regex=false).
    std::string pattern;
    auto take_split = [&](const glz::generic &step) {
        if (string_field(step, "type") != "Split") return;
        if (const auto *regex = field(step, "pattern"); regex) pattern = std::string(string_field(*regex, "Regex"));
    };
    take_split(file.pre_tokenizer);
    if (const auto *steps = field(file.pre_tokenizer, "pretokenizers"); steps && steps->is_array()) {
        for (const auto &step : steps->get_array()) {
            take_split(step);
            if (string_field(step, "type") == "ByteLevel") {
                const auto *use_regex = field(step, "use_regex");
                if (use_regex && use_regex->is_boolean() && use_regex->get_boolean()) {
                    return std::unexpected("unsupported pre-tokenizer: ByteLevel with use_regex");
                }
            }
        }
    }
    if (pattern.empty()) return std::unexpected("tokenizer.json has no Split pre-tokenizer");

    int error = 0;
    PCRE2_SIZE error_offset = 0;
    self->split = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(),
                                PCRE2_UTF | PCRE2_UCP | PCRE2_MATCH_INVALID_UTF, &error, &error_offset, nullptr);
    if (!self->split) {
        std::array<PCRE2_UCHAR, 256> message{};
        pcre2_get_error_message(error, message.data(), message.size());
        return std::unexpected("cannot compile the pre-tokenizer regex: " + std::string(reinterpret_cast<const char *>(message.data())));
    }
    pcre2_jit_compile(self->split, PCRE2_JIT_COMPLETE);

    // TemplateProcessing (alone or in a Sequence): special tokens around "A".
    auto take_template = [&](const glz::generic &processor) -> std::optional<std::string> {
        if (string_field(processor, "type") != "TemplateProcessing") return std::nullopt;
        const auto *single = field(processor, "single");
        const auto *specials = field(processor, "special_tokens");
        if (!single || !single->is_array()) return "TemplateProcessing without a single template";
        bool after = false;
        for (const auto &piece : single->get_array()) {
            if (field(piece, "Sequence")) {
                after = true;
                continue;
            }
            const auto *special = field(piece, "SpecialToken");
            if (!special) return "unknown template piece";
            const auto name = string_field(*special, "id");
            const auto *entry = specials ? field(*specials, name) : nullptr;
            const auto *ids = entry ? field(*entry, "ids") : nullptr;
            if (!ids || !ids->is_array()) return "template token without ids: " + std::string(name);
            for (const auto &id : ids->get_array()) {
                (after ? self->suffix : self->prefix).push_back(static_cast<i32>(id.get_number()));
            }
        }
        return std::string();
    };
    auto check = take_template(file.post_processor);
    if (const auto *processors = field(file.post_processor, "processors"); processors && processors->is_array()) {
        for (const auto &processor : processors->get_array()) {
            if (auto result = take_template(processor)) check = std::move(result);
        }
    }
    if (check && !check->empty()) return std::unexpected("unsupported post-processor: " + *check);

    return Tokenizer(std::move(self));
}

std::vector<i32> Tokenizer::encode(std::string_view text) const {
    auto ids = self->prefix;
    auto body = self->encode(text);
    ids.insert(ids.end(), body.begin(), body.end());
    ids.insert(ids.end(), self->suffix.begin(), self->suffix.end());
    return ids;
}

std::vector<i32> Tokenizer::encode_plain(std::string_view text) const { return self->encode(text); }

std::optional<i32> Tokenizer::token_id(std::string_view token) const {
    const std::string key(token);
    if (const auto it = self->added_ids.find(key); it != self->added_ids.end()) return it->second;
    if (const auto it = self->vocab.find(key); it != self->vocab.end()) return it->second;
    return std::nullopt;
}

} // namespace qvl::tokenizer
