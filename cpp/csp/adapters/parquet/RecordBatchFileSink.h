#ifndef _IN_CSP_ADAPTERS_PARQUET_RecordBatchFileSink_H
#define _IN_CSP_ADAPTERS_PARQUET_RecordBatchFileSink_H

#include <csp/adapters/parquet/RecordBatchSink.h>
#include <functional>
#include <string>

namespace csp::adapters::parquet
{

// Builds a RecordBatchSink that writes RecordBatches to files entirely in C++
// (no Python / no Arrow C Data Interface hop on the hot path).
//
// One sink == one logical output target:
//   - splitColumns == false : a single file (.parquet or .arrow) with all columns.
//   - splitColumns == true  : a directory; each column is written to its own file.
//
// File rotation is just onFileChange (close current, open next). compression /
// allowOverwrite / fileVisitor behave as on the Python sink. fileVisitor, if set,
// is invoked with the path of each file after it is closed.
RecordBatchSink makeFileSink( bool writeArrowBinary,
                              bool splitColumns,
                              const std::string & compression,
                              bool allowOverwrite,
                              std::function<void( const std::string & )> fileVisitor = {} );

} // namespace csp::adapters::parquet

#endif // _IN_CSP_ADAPTERS_PARQUET_RecordBatchFileSink_H
