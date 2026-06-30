/*
 * CSP exposes functionality through a list of types.
 * These are defined in CspType.h.
 *
 * Basic types need to be mapped to C types in an ABI-stable manner.
 *   - Unknown
 *   - Bool
 *   - Int/Unit 8/16/32/64
 *   - Double
 *
 * Complex types should be mapped into basic types where possible.
 *   - DateTime -> int64_t
 *   - TimeDelta -> int64_t
 *   - Date -> int32_t
 *   - Time -> int32_t
 *   - Enum -> int32_t (plus string mapping, if needed)
 *   - String -> char* + length
 *   - Struct -> opaque pointer + metadata
 *
 * Array types should be mapped to basic types if possible, otherwise to opaque pointer + metadata.
 *  - Array of Bool -> uint8_t* + length
 *  - Array of Int/Uint8/16/32/64 -> corresponding pointer + length
 *  - Array of Double -> double* + length
 *  - Array of DateTime/TimeDelta/Date/Time -> int64_t* / int32_t* + length
 *  - Array of Enum -> int32_t* + length (plus string mapping, if needed)
 *  - Array of String -> char** + length
 *  - Array of Struct -> opaque pointer* + length + metadata
 *  - Array of Array -> opaque pointer* + length + metadata
 *
 * DialectGenericType is a bit weird as we probably don't care about its internal structure.
 * For example, if its a PyObject*, we can just pass it as an opaque pointer as the other side
 * of the ABI boundary will need/know how to handle it based on the stability of the outer 
 * dialect itself.
 *
 * DialectGenericType -> opaque pointer + metadata
 */


#ifndef _IN_CSP_ENGINE_CCSPTYPE_H
#define _IN_CSP_ENGINE_CCSPTYPE_H

#ifdef __cplusplus
extern "C" {
#endif

    // Basic types
    typedef enum {
        CCSP_TYPE_UNKNOWN = 0,
        CCSP_TYPE_BOOL,
        CCSP_TYPE_INT8,
        CCSP_TYPE_UINT8,
        CCSP_TYPE_INT16,
        CCSP_TYPE_UINT16,
        CCSP_TYPE_INT32,
        CCSP_TYPE_UINT32,
        CCSP_TYPE_INT64,
        CCSP_TYPE_UINT64,
        CCSP_TYPE_DOUBLE,
        CCSP_TYPE_STRING,
        CCSP_TYPE_DATETIME,
        CCSP_TYPE_TIMEDELTA,
        CCSP_TYPE_DATE,
        CCSP_TYPE_TIME,
        CCSP_TYPE_ENUM,
        CCSP_TYPE_STRUCT,
        CCSP_TYPE_ARRAY,
        CCSP_TYPE_DIALECT_GENERIC
    } CCspType;
    

#ifdef __cplusplus
}
#endif

#endif
