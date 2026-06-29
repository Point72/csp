#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetOutputAdapter_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetOutputAdapter_H

#include <csp/adapters/parquet/ArrowSingleColumnArrayBuilder.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <string>

namespace csp::adapters::parquet { class ArrowBackedArrayBuilder; }

namespace csp::adapters::parquet
{
class ParquetWriter;

class ParquetOutputHandler
{
public:
    ParquetOutputHandler( ParquetWriter &parquetWriter )
            : m_parquetWriter( parquetWriter )
    {
    }

    virtual ~ParquetOutputHandler() {}

    uint32_t getChunkSize() const;

    virtual uint32_t getNumColumns() = 0;
    virtual std::shared_ptr<ArrowSingleColumnArrayBuilder> getColumnArrayBuilder( unsigned index ) = 0;
    virtual void writeValueFromTs( const TimeSeriesProvider *input ) = 0;

protected:
    ParquetWriter &m_parquetWriter;
};

class SingleColumnParquetOutputHandler : public ParquetOutputHandler
{
public:
    SingleColumnParquetOutputHandler( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &type,
                                      std::string columnName );

    uint32_t getNumColumns() override{ return 1; }

    std::shared_ptr<ArrowSingleColumnArrayBuilder> getColumnArrayBuilder( unsigned index ) override{ return m_columnArrayBuilder; };

    void writeValueFromTs( const TimeSeriesProvider *input ) override final
    {
        ( *m_valueHandler )( input );
    }

    // Write a value directly to the underlying column builder (used by dict basket writers)
    template< typename T, typename Ignored = void >
    void writeValue( const T & value );

protected:
    using ValueHandler = std::function<void( const TimeSeriesProvider * )>;

    std::unique_ptr<ValueHandler>                  m_valueHandler;
    std::shared_ptr<ArrowSingleColumnArrayBuilder> m_columnArrayBuilder;
};

class ListColumnParquetOutputHandler : public ParquetOutputHandler
{
public:
    ListColumnParquetOutputHandler( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &elemType, const std::string &columnName );

    uint32_t getNumColumns() override{ return 1; }

    std::shared_ptr<ArrowSingleColumnArrayBuilder> getColumnArrayBuilder( unsigned index ) override{ return m_columnArrayBuilder; };

    void writeValueFromTs( const TimeSeriesProvider *input ) override final
    {
        ( *m_valueHandler )( input );
    }

protected :
    using ValueHandler = std::function<void( const TimeSeriesProvider * )>;

    std::unique_ptr<ValueHandler>           m_valueHandler;
    std::shared_ptr<ListColumnArrayBuilder> m_columnArrayBuilder;
};

class SingleColumnParquetOutputAdapter final : public OutputAdapter, public SingleColumnParquetOutputHandler
{
public:
    SingleColumnParquetOutputAdapter( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &type, std::string columnName )
            : OutputAdapter( engine ), SingleColumnParquetOutputHandler( engine, parquetWriter, type, columnName )
    {
    }

    const char *name() const override{ return "ParquetSingleColumnOutputAdapter"; }

    void executeImpl() override;
};

class ListColumnParquetOutputAdapter : public OutputAdapter, public ListColumnParquetOutputHandler
{
public:
    ListColumnParquetOutputAdapter( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &type, std::string columnName )
            : OutputAdapter( engine ), ListColumnParquetOutputHandler( engine, parquetWriter, type, columnName )
    {
    }

    const char *name() const override{ return "ListColumnParquetOutputAdapter"; }

    void executeImpl() override;
};


class StructParquetOutputHandler : public ParquetOutputHandler
{
public:
    StructParquetOutputHandler( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &type, DictionaryPtr fieldMap );

    uint32_t getNumColumns() override{ return m_columnArrayBuilders.size(); }

    std::shared_ptr<ArrowSingleColumnArrayBuilder> getColumnArrayBuilder( unsigned index ) override
    {
        return m_columnArrayBuilders[ index ];
    }

    void writeValueFromTs( const TimeSeriesProvider *input ) override final;

protected:
    std::vector<ArrowBackedArrayBuilder *>                          m_arrowBuilders;
    std::vector<std::shared_ptr<ArrowSingleColumnArrayBuilder>>     m_columnArrayBuilders;
};

class StructParquetOutputAdapter final : public OutputAdapter, public StructParquetOutputHandler
{
public:
    StructParquetOutputAdapter( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &type, DictionaryPtr fieldMap )
            : OutputAdapter( engine ), StructParquetOutputHandler( engine, parquetWriter, type, fieldMap )
    {
    }

    const char *name() const override{ return "ParquetStructOutputAdapter"; }

    using StructParquetOutputHandler::StructParquetOutputHandler;
    void executeImpl() override;
};

}

#endif
