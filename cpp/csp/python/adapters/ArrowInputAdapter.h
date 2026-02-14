#ifndef _IN_CSP_ENGINE_ARROWINPUTADAPTER_H
#define _IN_CSP_ENGINE_ARROWINPUTADAPTER_H

#include <csp/engine/PullInputAdapter.h>
#include <csp/python/PyObjectPtr.h>
#include <csp/python/Conversions.h>
#include <csp/core/Time.h>
#include <Python.h>

#include <arrow/array.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/type.h>
#include <arrow/table.h>

#include <algorithm>
#include <memory>
#include <string>

namespace csp::python::arrow
{

class RecordBatchIterator
{
public:
    RecordBatchIterator() {}
    RecordBatchIterator( PyObjectPtr iter, std::shared_ptr<::arrow::Schema> schema ): m_iter( std::move( iter ) ), m_schema( schema )
    {
    }

    std::shared_ptr<::arrow::RecordBatch> next()
    {
        auto py_tuple = csp::python::PyObjectPtr::own( PyIter_Next( m_iter.get() ) );
        if( PyErr_Occurred() )
            CSP_THROW( csp::python::PythonPassthrough, "" );

        if( py_tuple.get() == NULL )
        {
            // No more data in the input steam
            return nullptr;
        }

        if( !PyTuple_Check( py_tuple.get() ) )
            CSP_THROW( csp::TypeError, "Invalid arrow data, expected tuple (using the PyCapsule C interface) got " << Py_TYPE( py_tuple.get() ) -> tp_name );

        auto num_elems = PyTuple_Size( py_tuple.get()  );
        if( num_elems != 2 )
            CSP_THROW( csp::TypeError, "Invalid arrow data, expected tuple (using the PyCapsule C interface) with 2 elements got " << num_elems );

        // Extract the record batch
        PyObject * py_array = PyTuple_GetItem( py_tuple.get(), 1 );
        if( !PyCapsule_IsValid( py_array, "arrow_array" ) )
            CSP_THROW( csp::TypeError, "Invalid arrow data, expected tuple from the PyCapsule C interface " );

        ArrowArray * c_array = reinterpret_cast<ArrowArray*>( PyCapsule_GetPointer( py_array, "arrow_array" ) );
        auto result = ::arrow::ImportRecordBatch( c_array, m_schema );
        if( !result.ok() )
            CSP_THROW( ValueError, "Failed to load record batches through PyCapsule C Data interface: " << result.status().ToString() );

        return result.ValueUnsafe();
    }

private:
    PyObjectPtr m_iter;
    std::shared_ptr<::arrow::Schema> m_schema;
};

void ReleaseArrowSchemaPyCapsule( PyObject * capsule )
{
    ArrowSchema * schema = reinterpret_cast<ArrowSchema*>( PyCapsule_GetPointer( capsule, "arrow_schema" ) );
    if ( schema -> release != NULL )
        schema -> release( schema );
    free( schema );
}

void ReleaseArrowArrayPyCapsule( PyObject * capsule )
{
    ArrowArray * array = reinterpret_cast<ArrowArray*>( PyCapsule_GetPointer( capsule, "arrow_array" ) );
    if ( array -> release != NULL )
        array -> release( array );
    free( array );
}

class RecordBatchInputAdapter: public PullInputAdapter<std::vector<DialectGenericType>>
{
public:
    RecordBatchInputAdapter( Engine * engine, CspTypePtr & type, PyObjectPtr pySchema, std::string tsColName, PyObjectPtr source, bool expectSmallBatches )
        : PullInputAdapter<std::vector<DialectGenericType>>( engine, type, PushMode::LAST_VALUE ),
          m_tsColName( tsColName ),
          m_expectSmallBatches( expectSmallBatches ),
          m_finished( false )
    {
        // Extract the arrow schema
        ArrowSchema * c_schema = reinterpret_cast<ArrowSchema*>( PyCapsule_GetPointer( pySchema.get(), "arrow_schema" ) );
        auto result = ::arrow::ImportSchema( c_schema );
        if( !result.ok() )
            CSP_THROW( ValueError, "Failed to load schema for record batches through the PyCapsule C Data interface: " << result.status().ToString() );
        m_schema = std::move( result.ValueUnsafe() );

        auto tsField = m_schema -> GetFieldByName( m_tsColName );
        auto timestampType = std::static_pointer_cast<::arrow::TimestampType>( tsField -> type() );
        switch( timestampType -> unit() )
        {
            case ::arrow::TimeUnit::SECOND:
                m_multiplier = csp::NANOS_PER_SECOND;
                break;
            case ::arrow::TimeUnit::MILLI:
                m_multiplier = csp::NANOS_PER_MILLISECOND;
                break;
            case ::arrow::TimeUnit::MICRO:
                m_multiplier = csp::NANOS_PER_MICROSECOND;
                break;
            case ::arrow::TimeUnit::NANO:
                m_multiplier = 1;
                break;
            default:
                CSP_THROW( ValueError, "Unsupported unit type for arrow timestamp column" );
        }

        m_source = RecordBatchIterator( source, m_schema );
    }

    int64_t findFirstMatchingIndex()
    {
        // Find the first index with time equal or greater than `time`
        auto first_time = m_tsArray -> Value( 0 );
        if( first_time >= m_startTime )
        {
            // Early break
            return 0;
        }

        auto last_time = m_tsArray -> Value( m_numRows - 1 );
        if( last_time < m_startTime )
        {
            // Early break
            return m_numRows;
        }
        ::arrow::TimestampArray::IteratorType it;
        auto begin = ++( m_tsArray -> begin() );
        if( m_expectSmallBatches )
        {
            auto predicate = [this]( std::optional<int64_t> new_time ) -> bool { return new_time.value() >= this -> m_startTime; };
            it = std::find_if( begin, m_endIt, predicate );
        }
        else
            it = std::lower_bound( begin, m_endIt, m_startTime );
        return it.index();
    }

    int64_t findNextLargerTimestampIndex( int64_t start_idx )
    {
        // Find the first index with time just greater than the time at start_idx
        auto cur_time = m_tsArray -> Value( start_idx );
        auto begin = ::arrow::TimestampArray::IteratorType( *m_tsArray, start_idx );
        ::arrow::TimestampArray::IteratorType it;
        if( m_expectSmallBatches )
        {
            auto predicate = [cur_time]( std::optional<int64_t> new_time ) -> bool { return cur_time != new_time.value(); };
            it = std::find_if( begin, m_endIt, predicate );
        }
        else
        {
            if( m_arrayLastTime == cur_time )
                return m_numRows;
            it = std::upper_bound( begin, m_endIt, cur_time );
        }
        return it.index();
    }

    std::shared_ptr<::arrow::RecordBatch> updateStateFromNextRecordBatch()
    {
        std::shared_ptr<::arrow::RecordBatch> rb;
        while( ( rb = m_source.next() ) && ( rb -> num_rows() == 0 ) ) {}
        if( rb )
        {
            auto array = rb -> GetColumnByName( m_tsColName );
            if( !array )
                CSP_THROW( ValueError, "Failed to get timestamp column " << m_tsColName << " from record batch " << rb -> ToString() );

            m_tsArray = std::static_pointer_cast<::arrow::TimestampArray>( array );
            m_numRows = m_tsArray -> length();
            m_endIt = m_tsArray -> end();
            m_arrayLastTime = m_tsArray -> Value( m_numRows - 1 );
        }
        m_curRecordBatch = rb;
        return rb;
    }

    void start( DateTime start, DateTime end ) override
    {
        // start and end as multiples of the unit in timestamp column
        auto start_nanos = start.asNanoseconds();
        m_startTime = ( start_nanos % m_multiplier == 0 ) ? start_nanos / m_multiplier : start_nanos / m_multiplier + 1;
        m_endTime = end.asNanoseconds() / m_multiplier;

        // Find the starting index where time >= start
        while( !m_finished )
        {
            updateStateFromNextRecordBatch();
            if( !m_curRecordBatch )
            {
                m_finished = true;
                break;
            }
            m_curBatchIdx = findFirstMatchingIndex();
            if( m_curBatchIdx < m_numRows )
                break;
        }
        PullInputAdapter<std::vector<DialectGenericType>>::start( start, end );
    }

    DialectGenericType convertRecordBatchToPython( std::shared_ptr<::arrow::RecordBatch> rb )
    {
        ArrowSchema* rb_schema = ( ArrowSchema* )malloc( sizeof( ArrowSchema ) );
        ArrowArray* rb_array = ( ArrowArray* )malloc( sizeof( ArrowArray ) );
        ::arrow::Status st = ::arrow::ExportRecordBatch( *rb, rb_array, rb_schema );
        auto py_schema = csp::python::PyObjectPtr::own( PyCapsule_New( rb_schema, "arrow_schema", ReleaseArrowSchemaPyCapsule ) );
        auto py_array = csp::python::PyObjectPtr::own( PyCapsule_New( rb_array, "arrow_array", ReleaseArrowArrayPyCapsule ) );
        auto py_tuple = csp::python::PyObjectPtr::own( PyTuple_Pack( 2, py_schema.get(), py_array.get() ) );
        return csp::python::fromPython<DialectGenericType>( py_tuple.get() );
    }

    bool next( DateTime & t, std::vector<DialectGenericType> & value ) override
    {
        std::vector<DialectGenericType> cur_result;
        int64_t cur_ts = 0;
        while( !m_finished )
        {
            // Slice current record batch
            auto new_ts = m_tsArray -> Value( m_curBatchIdx );
            if( new_ts > m_endTime )
            {
                // Past the end time
                m_finished = true;
                break;
            }
            if( !cur_result.empty() && new_ts != cur_ts )
            {
                // Next timestamp encountered, return the current list of record batches
                break;
            }
            cur_ts = new_ts;
            auto next_idx = findNextLargerTimestampIndex( m_curBatchIdx );
            auto slice = m_curRecordBatch -> Slice( m_curBatchIdx, next_idx - m_curBatchIdx );
            cur_result.emplace_back( convertRecordBatchToPython( slice ) );
            m_curBatchIdx = next_idx;
            if( m_curBatchIdx != m_numRows )
            {
                // All rows for current timestamp have been found
                break;
            }
            // Get the next record batch
            updateStateFromNextRecordBatch();
            if( !m_curRecordBatch )
            {
                m_finished = true;
                break;
            }
            m_curBatchIdx = 0;
        }
        if( !cur_result.empty() )
        {
            value = std::move( cur_result );
            t = csp::DateTime::fromNanoseconds( cur_ts * m_multiplier );
            return true;
        }
        return false;
    }

private:
    std::string m_tsColName;
    RecordBatchIterator m_source;

    int m_expectSmallBatches;
    bool m_finished;
    std::shared_ptr<::arrow::Schema> m_schema;
    std::shared_ptr<::arrow::RecordBatch> m_curRecordBatch;
    std::shared_ptr<::arrow::TimestampArray> m_tsArray;
    ::arrow::TimestampArray::IteratorType m_endIt;
    int64_t m_arrayLastTime;
    int64_t m_multiplier, m_numRows, m_startTime, m_endTime, m_curBatchIdx;
};

};

#endif
