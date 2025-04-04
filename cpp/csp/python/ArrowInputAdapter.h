#ifndef _IN_CSP_ENGINE_ARROWINPUTADAPTER_H
#define _IN_CSP_ENGINE_ARROWINPUTADAPTER_H

#include <csp/engine/PullInputAdapter.h>
#include <csp/python/PyObjectPtr.h>
#include <csp/python/Conversions.h>
#include <Python.h>

#include <arrow/array.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/type.h>
#include <arrow/table.h>

#include <memory>
#include <string>

namespace csp::python::arrow
{

class RecordBatchIterator
{
public:
    RecordBatchIterator( PyObject * iter, PyObject * py_schema ): m_iter( PyObjectPtr::incref( iter ) )
    {
        // Extract the arrow schema
        struct ArrowSchema * c_schema = reinterpret_cast<struct ArrowSchema*>( PyCapsule_GetPointer( py_schema, "arrow_schema" ) );
        auto result = ::arrow::ImportSchema( c_schema );
        if( !result.ok() )
            CSP_THROW( ValueError, "Failed to load schema for record batches through the PyCapsule C Data interface: " << result.status().ToString() );
        m_schema = std::move(result).ValueUnsafe();
    }

    std::shared_ptr<::arrow::RecordBatch> next()
    {
        auto py_tuple = csp::python::PyObjectPtr::own( PyIter_Next( m_iter.get() ) );
        if( py_tuple.get() == NULL )
        {
            // No more data in the input steam
            return nullptr;
        }
        else
        {
            // Extract the record batch
            PyObject * py_array = PyTuple_GET_ITEM( py_tuple.get(), 1 );
            struct ArrowArray * c_array = reinterpret_cast<struct ArrowArray*>( PyCapsule_GetPointer( py_array, "arrow_array" ) );
            auto result = ::arrow::ImportRecordBatch( c_array, m_schema );
            if( !result.ok() )
                CSP_THROW( ValueError, "Failed to load record batches through PyCapsule C Data interface: " << result.status().ToString() );
            return std::move(result).ValueUnsafe();
        }
    }

private:
    PyObjectPtr m_iter;
    std::shared_ptr<::arrow::Schema> m_schema;
};

void ReleaseArrowSchemaPyCapsule( PyObject * capsule ) {
    struct ArrowSchema * schema = reinterpret_cast<struct ArrowSchema*>( PyCapsule_GetPointer( capsule, "arrow_schema" ) );
    if ( schema -> release != NULL )
    {
        schema -> release( schema );
    }
    free( schema );
}

void ReleaseArrowArrayPyCapsule( PyObject * capsule ) {
    struct ArrowArray * array = reinterpret_cast<struct ArrowArray*>( PyCapsule_GetPointer( capsule, "arrow_array" ) );
    if ( array -> release != NULL ) {
        array -> release( array );
    }
    free( array );
}

class RecordBatchInputAdapter: public PullInputAdapter<std::vector<DialectGenericType>>
{
public:
    RecordBatchInputAdapter( Engine * engine, CspTypePtr & type, std::string tsColName, RecordBatchIterator source, int expectSmallBatches )
        : PullInputAdapter<std::vector<DialectGenericType>>( engine, type, PushMode::LAST_VALUE ),
          m_tsColName( tsColName ),
          m_source( source ),
          m_expectSmallBatches( expectSmallBatches != 0 ),
          m_finished( false )
    {
    }

    int64_t findFirstMatchingIndex( DateTime time )
    {
        // Find the first index with time equal or greater than `time`
        auto m_numRows = m_tsArray -> length();
        auto start_time = ( time.asNanoseconds() % m_multiplier == 0 ) ? time.asNanoseconds()/m_multiplier : time.asNanoseconds()/m_multiplier + 1;

        auto first_time = m_tsArray -> Value( 0 );
        if( first_time >= start_time )
        {
            return 0;
        }

        auto last_time = m_tsArray -> Value( m_numRows - 1 );
        if( last_time < start_time )
        {
            return -1;
        }

        auto l = 0;
        auto r = m_numRows-1;
        auto mid = 0;
        while( l <= r )
        {
            mid = (l + r) / 2;
            auto mid_time = m_tsArray -> Value( mid );
            if( mid_time < start_time )
            {
                auto mid_next_time = m_tsArray -> Value( mid + 1 );
                if( mid_next_time >= start_time )
                {
                    break;
                }
                else
                {
                    l = mid+1;
                }
            }
            else if ( mid_time > start_time )
            {
                r = mid - 1;
            }
        }
        return mid+1;
    }


    int64_t findNextLargerTimestampIndex( int64_t start_idx )
    {
        // Find the first index with time just greater than the time at start_idx
        int64_t res = 0;
        auto cur_time = m_tsArray -> Value( start_idx );
        if( m_expectSmallBatches )
        {
            auto idx = start_idx + 1;
            while( idx < m_numRows && m_tsArray -> Value( idx ) == cur_time )
            {
                idx++;
            }
            res = idx;
        }
        else
        {
            auto last_time = m_tsArray -> Value( m_numRows - 1 );
            if( last_time == cur_time )
            {
                return m_numRows;
            }

            auto l = start_idx;
            auto r = m_numRows-1;
            auto mid = 0;
            while( l <= r )
            {
                mid = (l + r) / 2;
                auto mid_time = m_tsArray -> Value( mid );
                if( mid_time == cur_time )
                {
                    auto mid_next_time = m_tsArray -> Value( mid + 1 );
                    if( mid_next_time > cur_time )
                    {
                        break;
                    }
                    else
                    {
                        l = mid+1;
                    }
                }
                else if ( mid_time > cur_time )
                {
                    r = mid - 1;
                }
            }
            res = mid+1;
        }
        return res;
    }

    void start( DateTime start, DateTime end ) override
    {
        // Find the starting index where time >= start
        m_endTime = end.asNanoseconds();
        bool reachedStartTime = false;
        while( !reachedStartTime and !m_finished )
        {
            m_curRecordBatch = getNonEmptyRecordBatchFromSource();
            if( !m_curRecordBatch )
            {
                m_finished = true;
                continue;
            }
            auto schema = m_curRecordBatch -> schema();
            auto tsField = schema -> GetFieldByName( m_tsColName );
            auto timestampType = std::static_pointer_cast<::arrow::TimestampType>( tsField -> type() );
            auto array = m_curRecordBatch -> GetColumnByName( m_tsColName );
            if( !array )
            {
                m_finished = true;
                continue;
            }

            m_tsArray = std::static_pointer_cast<::arrow::TimestampArray>( array );
            m_numRows = m_tsArray -> length();

            switch( timestampType -> unit() )
            {
                case ::arrow::TimeUnit::SECOND:
                {
                    m_multiplier = 1000000000;
                    break;
                }
                case ::arrow::TimeUnit::MILLI:
                {
                    m_multiplier = 1000000;
                    break;
                }
                case ::arrow::TimeUnit::MICRO:
                {
                    m_multiplier = 1000;
                    break;
                }
                case ::arrow::TimeUnit::NANO:
                {
                    m_multiplier = 1;
                    break;
                }
                default:
                {
                    CSP_THROW( ValueError, "Unsupported unit type for arrow timestamp column" );
                }
            }
            m_curBatchIdx = findFirstMatchingIndex( start );
            if( m_curBatchIdx >= 0 )
            {
                break;
            }
        }
        PullInputAdapter<std::vector<DialectGenericType>>::start( start, end );
    }

    std::shared_ptr<::arrow::RecordBatch> getNonEmptyRecordBatchFromSource()
    {
        std::shared_ptr<::arrow::RecordBatch> rb;
        while( ( rb = m_source.next() ) && ( rb -> num_rows() == 0) ) { continue; }
        return rb;
    }

    DialectGenericType convertRecordBatchToPython( std::shared_ptr<::arrow::RecordBatch> rb )
    {
        struct ArrowSchema* rb_schema = ( struct ArrowSchema* )malloc( sizeof( struct ArrowSchema ) );
        struct ArrowArray* rb_array = ( struct ArrowArray* )malloc( sizeof( struct ArrowArray ) );
        ::arrow::Status st = ::arrow::ExportRecordBatch( *rb, rb_array, rb_schema );
        auto py_schema = csp::python::PyObjectPtr::own( PyCapsule_New( rb_schema, "arrow_schema", ReleaseArrowSchemaPyCapsule ) );
        auto py_array = csp::python::PyObjectPtr::own( PyCapsule_New( rb_array, "arrow_array", ReleaseArrowArrayPyCapsule ) );
        auto py_tuple = csp::python::PyObjectPtr::own( PyTuple_Pack( 2, py_schema.get(), py_array.get() ) );
        return csp::python::fromPython<DialectGenericType>( py_tuple.get() );
    }

    bool next( DateTime & t, std::vector<DialectGenericType> & value ) override
    {
        m_curResult.clear();
        bool newRecordBatch = false;
        while( !m_finished )
        {
            // Slice current record batch
            auto new_ts = m_tsArray -> Value( m_curBatchIdx );
            if( new_ts * m_multiplier > m_endTime )
            {
                // Past the end time
                m_finished = true;
                break;
            }
            if( newRecordBatch && new_ts != m_curTs )
            {
                // Next timestamp encountered, return the current list of record batches
                value = m_curResult;
                m_time = csp::DateTime::fromNanoseconds( m_curTs * m_multiplier );
                t = m_time;
                return true;
            }
            m_curTs = new_ts;
            auto next_idx = findNextLargerTimestampIndex( m_curBatchIdx );
            auto slice = m_curRecordBatch -> Slice( m_curBatchIdx, next_idx - m_curBatchIdx );
            m_curResult.emplace_back( convertRecordBatchToPython( slice ) );
            m_curBatchIdx = next_idx;
            if( m_curBatchIdx != m_numRows )
            {
                // All rows for current timestamp have been found
                value = m_curResult;
                m_time = csp::DateTime::fromNanoseconds( m_curTs * m_multiplier );
                t = m_time;
                return true;
            }
            // Get the next record batch
            m_curRecordBatch = getNonEmptyRecordBatchFromSource();
            if( !m_curRecordBatch )
            {
                m_finished = true;
                break;
            }
            auto array = m_curRecordBatch -> GetColumnByName( m_tsColName );
            m_tsArray = std::static_pointer_cast<::arrow::TimestampArray>( array );
            m_numRows = m_tsArray -> length();
            m_curBatchIdx = 0;
            newRecordBatch = true;
        }
        if( !m_curResult.empty() )
        {
            value = m_curResult;
            m_time = csp::DateTime::fromNanoseconds( m_curTs * m_multiplier );
            t = m_time;
            return true;
        }
        return false;
    }

private:
    std::string m_tsColName;
    RecordBatchIterator m_source;
    int m_expectSmallBatches;
    bool m_finished;
    std::shared_ptr<::arrow::RecordBatch> m_curRecordBatch;
    std::shared_ptr<::arrow::TimestampArray> m_tsArray;
    int64_t m_multiplier, m_numRows, m_curTs, m_endTime, m_curBatchIdx;
    std::vector<DialectGenericType> m_curResult;
    DateTime m_time;
};

};

#endif
