#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetOutputFilenameAdapter_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetOutputFilenameAdapter_H

#include <csp/engine/OutputAdapter.h>


namespace csp::adapters::parquet
{
class ParquetOutputAdapterManager;

class ParquetOutputFilenameAdapter : public csp::OutputAdapter
{
public:
    ParquetOutputFilenameAdapter( Engine *engine, ParquetOutputAdapterManager &parquetOutputAdapterManager )
            : csp::OutputAdapter( engine ), m_parquetOutputAdapterManager( parquetOutputAdapterManager )
    {
    }

    void executeImpl() override;

    const char *name() const override { return "ParquetOutputFilenameAdapter"; }

protected:
    ParquetOutputAdapterManager &m_parquetOutputAdapterManager;
};

}

#endif
