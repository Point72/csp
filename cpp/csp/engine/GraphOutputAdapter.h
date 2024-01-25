#ifndef _IN_CSP_ENGINE_GRAPHOUTPUTADAPTER_H
#define _IN_CSP_ENGINE_GRAPHOUTPUTADAPTER_H

#include <csp/engine/OutputAdapter.h>
#include <memory>

namespace csp
{

//GraphOutputAdapter - latches onto a ts for use at the end of csp.run calls to convert the buffered data into dialect readable form
//Note some nuances with GraphOutputAdapter as it pertains to dynamic graphs.  csp.add_graph_output calls in dynamic graphs have an interesting problem in
//that the timeseries they latch onto can be destroyed before the end of the engine run.  For this reason, processResults() can be called before
//the end of a csp.run call to pre-process the data before the time series input is released.  
//GraphOutputAdapters are also special in that they are registered in both a dynamic engine and the root engine as sahred_ptr.  They make it into root
//so that the root processes them at the end of the csp.run call ( dynamics could be shutdown by then ).  We also register in root
//to ensure we dont hit key clashses.
class GraphOutputAdapter : public OutputAdapter, public std::enable_shared_from_this<GraphOutputAdapter>
{
public:
    GraphOutputAdapter( csp::Engine * engine, int32_t tickCount, TimeDelta tickHistory );
    ~GraphOutputAdapter();

    const char * name() const override;

    void start() override;
    void stop() override;
    void executeImpl() override {}

    int32_t tickCount() const     { return m_tickCount; }
    TimeDelta tickHistory() const { return m_tickHistory; }

private:
    virtual void processResults() = 0;

    TimeDelta m_tickHistory;
    int32_t   m_tickCount;
};

}

#endif
