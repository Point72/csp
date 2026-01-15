#ifndef _IN_CSP_ADAPTERS_UTILS_AVROINCLUDES_H
#define _IN_CSP_ADAPTERS_UTILS_AVROINCLUDES_H

// =============================================================================
// Workaround for avro-cpp fmt::formatter incompatibility on Windows
// =============================================================================
// conda-forge's avro-cpp (as of early 2025) has outdated fmt::formatter
// specializations with non-const format() methods, but fmt v11+ requires const.
//
// The avro formatters are guarded by:
//   #if FMT_VERSION >= 90000
//
// We temporarily set FMT_VERSION below this threshold to prevent avro from
// defining its broken formatters, then restore it and define const-correct
// versions.
//
// This workaround can be removed once conda-forge updates avro-cpp.
// =============================================================================

#include <fmt/format.h>

#ifdef _MSC_VER
#pragma push_macro("FMT_VERSION")
#undef FMT_VERSION
#define FMT_VERSION 80000
#endif

// Include all avro headers that may be needed
#include <avro/Compiler.hh>
#include <avro/Decoder.hh>
#include <avro/Encoder.hh>
#include <avro/Generic.hh>
#include <avro/GenericDatum.hh>
#include <avro/Node.hh>
#include <avro/Schema.hh>
#include <avro/Stream.hh>
#include <avro/ValidSchema.hh>

#ifdef _MSC_VER
#undef FMT_VERSION
#pragma pop_macro("FMT_VERSION")

// Define const-correct fmt::formatter specializations for avro types
// These mirror the definitions in avro's headers but with the required const

template<>
struct fmt::formatter<avro::Name> : fmt::formatter<std::string> {
    auto format(const avro::Name &n, fmt::format_context &ctx) const {
        return fmt::format_to(ctx.out(), "{}", n.fullname());
    }
};

template<>
struct fmt::formatter<avro::Type> : fmt::formatter<std::string> {
    auto format(avro::Type t, fmt::format_context &ctx) const {
        return fmt::format_to(ctx.out(), "{}", avro::toString(t));
    }
};

#endif // _MSC_VER

#endif // _IN_CSP_ADAPTERS_UTILS_AVROINCLUDES_H
