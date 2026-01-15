#ifndef _IN_CSP_ADAPTERS_UTILS_AVROINCLUDES_H
#define _IN_CSP_ADAPTERS_UTILS_AVROINCLUDES_H

// Centralized avro includes.
// On Windows conda-forge builds, headers are patched at CMake configure time
// (see FindAvro.cmake) to fix fmt v12 compatibility.

#include <avro/Compiler.hh>
#include <avro/Decoder.hh>
#include <avro/Encoder.hh>
#include <avro/Generic.hh>
#include <avro/GenericDatum.hh>
#include <avro/Node.hh>
#include <avro/Schema.hh>
#include <avro/Stream.hh>
#include <avro/ValidSchema.hh>

#endif
