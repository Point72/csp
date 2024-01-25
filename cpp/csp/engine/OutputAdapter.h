#ifndef _IN_CSP_ENGINE_OUTPUTADAPTER_H
#define _IN_CSP_ENGINE_OUTPUTADAPTER_H

#include <csp/engine/Consumer.h>
#include <csp/engine/Engine.h>

namespace csp
{

class TimeSeriesProvider;

class OutputAdapter : public Consumer, public EngineOwned
{
public:
    OutputAdapter( csp::Engine * engine );
    ~OutputAdapter();

    TimeSeriesProvider * input() const { return m_input; }

    void link( TimeSeriesProvider * input );

    input_iterator inputs() const override { return input_iterator( &m_input ); }

private:
    //not owned
    TimeSeriesProvider * m_input;

};

}

#endif
