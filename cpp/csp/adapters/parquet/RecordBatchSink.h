#ifndef _IN_CSP_ADAPTERS_PARQUET_RecordBatchSink_H
#define _IN_CSP_ADAPTERS_PARQUET_RecordBatchSink_H

#include <arrow/record_batch.h>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace arrow { class Schema; }

namespace csp::adapters::parquet
{

// Writes the RecordBatches produced by ParquetWriter to files, entirely in C++ (no Python / no Arrow
// C Data Interface hop on the hot path).
//
// One sink == one logical output target:
//   - splitColumns == false : a single file (.parquet or .arrow) with all columns.
//   - splitColumns == true  : a directory; each column is written to its own file.
//
// The writer drives the lifecycle: onStart() once with the schema, then onFileChange()/onBatch() as
// data flows, then onStop() at teardown. File rotation is just onFileChange (close current, open
// next). fileVisitor, if set, is invoked with the path of each file after it is closed.
class RecordBatchSink
{
public:
    RecordBatchSink( bool writeArrowBinary,
                     bool splitColumns,
                     std::string compression,
                     bool allowOverwrite,
                     std::function<void( const std::string & )> fileVisitor = {},
                     bool useDictionary = true );

    // Called once with the output schema before any batches.
    void onStart( const std::shared_ptr<::arrow::Schema> & schema );
    // Called with each full RecordBatch. A batch only arrives while a file is open.
    void onBatch( const std::shared_ptr<::arrow::RecordBatch> & rb );
    // Close the current file (if any) and open path (if non-empty). Returns true if a file is open
    // after the call (path was non-empty), false otherwise.
    bool onFileChange( const std::string & path );
    // Close the current file (if any).
    void onStop();

private:
    // A single open output file: write one batch, then close. Type-erased over format so the rest
    // of the sink never branches on parquet-vs-ipc-vs-split.
    struct FileWriter
    {
        std::function<void( const std::shared_ptr<::arrow::RecordBatch> & )> writeBatch;
        std::function<void()>                                                close;
    };

    // Format/layout builders. Kept as plain functions (not a class hierarchy): each returns a
    // FileWriter that writes one open file. They are static members (rather than .cpp-local free
    // functions) only so they can name the private FileWriter type.
    static FileWriter makeParquetFileWriter( const std::string & path, const std::shared_ptr<::arrow::Schema> & schema,
                                             const std::string & compression, bool allowOverwrite, bool useDictionary );
    static FileWriter makeIpcFileWriter( const std::string & path, const std::shared_ptr<::arrow::Schema> & schema,
                                         const std::string & compression, bool allowOverwrite );
    static FileWriter makeSplitWriter( const std::string & dir, const std::shared_ptr<::arrow::Schema> & schema,
                                       bool writeArrowBinary, const std::string & compression, bool allowOverwrite,
                                       bool useDictionary, const std::string & extension );

    // (path) -> open FileWriter for m_schema, honoring this sink's format/layout/compression settings.
    FileWriter openWriter( const std::string & path );
    // Close the current file (if any), running fileVisitor after, exactly once.
    void closeCurrent();

    const bool                                  m_writeArrowBinary;
    const bool                                  m_splitColumns;
    const std::string                           m_compression;
    const bool                                  m_allowOverwrite;
    std::function<void( const std::string & )>  m_fileVisitor;
    const bool                                  m_useDictionary;
    const std::string                           m_extension;

    std::shared_ptr<::arrow::Schema>            m_schema;
    std::optional<FileWriter>                   m_current;
    std::string                                 m_currentPath;
};

} // namespace csp::adapters::parquet

#endif // _IN_CSP_ADAPTERS_PARQUET_RecordBatchSink_H
