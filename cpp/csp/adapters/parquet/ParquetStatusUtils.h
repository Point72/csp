#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetStatusUtils_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetStatusUtils_H

#include <csp/core/Exception.h>

#define STATUS_OK_OR_THROW_RUNTIME( EXPR, MESSAGE )                                                         \
    do                                                                                                      \
    {                                                                                                       \
        arrow::Status st = EXPR;                                                                            \
        CSP_TRUE_OR_THROW_RUNTIME( st.ok(), MESSAGE << ':' << st.ToString());                               \
    } while( 0 )

#endif
