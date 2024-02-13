#ifndef _IN_CSP_ENGINE_DYNAMICENGINE_H
#define _IN_CSP_ENGINE_DYNAMICENGINE_H

#include <csp/engine/Engine.h>
#include <csp/engine/OutputAdapter.h>
#include <string>
#include <unordered_map>

namespace csp
{

class DynamicEngine final : public Engine
{
public:
    using ShutdownFn = std::function<void()>;
    DynamicEngine( CycleStepTable & stepTable, RootEngine * root, ShutdownFn && shutdownFn );

    virtual ~DynamicEngine() {}

    class GraphOutput : public OutputAdapter
    {
    public:
        GraphOutput( Engine * engine );
        ~GraphOutput();

        void start() override;
        void executeImpl() override {}
        const char * name() const override { return "DynamicEngine::GraphOutput"; }
    };

    void registerGraphOutput( const std::string & key, GraphOutput * output );
    TimeSeriesProvider * outputTs( const std::string & key );

    void shutdown();

private:
    //these are outputs returned from the sub-graph, as opposed to csp.add_graph_output calls
    using GraphOutputs = std::unordered_map<std::string,GraphOutput *>;
    GraphOutputs m_graphOutputs;
    ShutdownFn   m_shutdownFn;
};

}

#endif
