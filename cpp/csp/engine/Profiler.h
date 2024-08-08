#ifndef _IN_CSP_ENGINE_PROFILER_H
#define _IN_CSP_ENGINE_PROFILER_H

#include <csp/core/Platform.h>
#include <csp/core/Time.h>
#include <csp/engine/Dictionary.h>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <stack>
#include <vector>

namespace csp
{

class Profiler
{
    // All times are returned as doubles representing time in seconds

    public:
    
        Profiler() = default;

        ~Profiler() 
        {
            if( m_cycleFile )
                m_cycleFile.close();
            if( m_nodeFile )
                m_nodeFile.close();
        }
        
        void startNode()
        { 
            m_callStack.push( DateTime::now() );
        }

        void finishNode( const std::string & name )
        {
            TimeDelta t = DateTime::now() - m_callStack.top();
            m_nodeStats[name].add_exec( t );
            if( m_nodeFile )
                m_nodeFile << name << "," << toSeconds( t ) << std::endl;
            m_callStack.pop();
        }

        DictionaryPtr totalNodeTimes() const
        {
            Dictionary time_by_node;
            for( auto it = m_nodeStats.begin(); it != m_nodeStats.end(); ++it )
                time_by_node.insert<double>( it -> first, toSeconds( it -> second.m_totalTime ) );

            return std::make_shared<Dictionary>( std::move( time_by_node ) );
        }

        DictionaryPtr maxNodeTimes() const
        {
            Dictionary max_by_node;
            for( auto it = m_nodeStats.begin(); it != m_nodeStats.end(); ++it )
                max_by_node.insert<double>( it -> first, toSeconds( it -> second.m_maxTime ) );

            return std::make_shared<Dictionary>( std::move( max_by_node ) );
        }

        DictionaryPtr countNodeExecs() const
        {
            Dictionary exec_by_node;
            for( auto it = m_nodeStats.begin(); it != m_nodeStats.end(); ++it )
                exec_by_node.insert<int64_t>( it -> first, it -> second.m_exec );

            return std::make_shared<Dictionary>( std::move( exec_by_node ) );
        }

        DictionaryPtr allNodeData() const
        {
            Dictionary data_by_node;
            for( auto it = m_nodeStats.begin(); it != m_nodeStats.end(); ++it )
            {
                Dictionary d;
                d.insert<int64_t>( "executions", it -> second.m_exec );
                d.insert<double>( "max_time", toSeconds( it -> second.m_maxTime ) );
                d.insert<double>( "total_time", toSeconds( it -> second.m_totalTime ) );
                data_by_node.insert<DictionaryPtr>( it -> first, std::make_shared<Dictionary>( d ) );
            }

            return std::make_shared<Dictionary>( std::move( data_by_node ) );
        }

        void startCycle()
        {
            m_callStack.push( DateTime::now() );
        }

        void finishCycle() 
        {
            TimeDelta t = DateTime::now() - m_callStack.top();
            m_cycleStat.add_exec( t );
            if( m_cycleFile )
                m_cycleFile << toSeconds( t ) << std::endl;
            m_callStack.pop();
        }
        
        double averageCycleTime() const 
        { 
            return toSeconds( m_cycleStat.m_totalTime ) / m_cycleStat.m_exec; 
        }

        double maxCycleTime() const 
        { 
            return toSeconds( m_cycleStat.m_maxTime ); 
        }

        double utilization( int64_t nodes ) const
        {
            auto accum = []( int64_t a, const std::pair<std::string, ProfStat> & b ) { return a + b.second.m_exec; };
            return static_cast<double>( std::accumulate( m_nodeStats.begin(), m_nodeStats.end(), 0, accum ) ) / ( m_cycleStat.m_exec * ( nodes - 2 ) ); // don't include profiler node and its associated nullts
        }

        DictionaryPtr getAllStats( int64_t nodes ) const
        {
            Dictionary d;
            // Insert cycle count and cycle times
            d.insert<int64_t>( "cycle_count", m_cycleStat.m_exec );
            d.insert<double>( "average_cycle_time", averageCycleTime() );
            d.insert<double>( "max_cycle_time", maxCycleTime() );

            // Insert node-level stats and utilization
            d.insert<DictionaryPtr>( "node_stats", allNodeData() );
            d.insert<double>( "utilization", utilization( nodes ) );

            return std::make_shared<Dictionary>( std::move( d ) );
        }

        void use_prof_file( const std::string & fname, bool node )
        {
            if( node )
            {
                m_nodeFile.open( fname, std::ofstream::out );
                if( !m_nodeFile.is_open() )
                    CSP_THROW( ValueError, "Cannot open file due to invalid path: " << fname );
                m_nodeFile << "Node Type,Execution Time" << std::endl; 
            }
            else
            {
                m_cycleFile.open( fname, std::ofstream::out );
                if( !m_cycleFile.is_open() )
                    CSP_THROW( ValueError, "Cannot open file due to invalid path: " << fname );
                m_cycleFile << "Execution Time" << std::endl;
            }
        }

    private:

        struct ProfStat
        {
            int64_t m_exec = 0;
            TimeDelta m_maxTime = TimeDelta::ZERO();
            TimeDelta m_totalTime = TimeDelta::ZERO();

            void add_exec( const TimeDelta & t )
            {
                m_exec++;
                m_maxTime = std::max( m_maxTime, t );
                m_totalTime += t;
            }
        };

        inline double toSeconds( const TimeDelta & t ) const { return static_cast<double>( t.asNanoseconds() ) / NANOS_PER_SECOND; }

        std::unordered_map<std::string, ProfStat> m_nodeStats;
        std::stack<DateTime> m_callStack;
        ProfStat m_cycleStat;

        std::ofstream m_cycleFile;
        std::ofstream m_nodeFile;
};

}

#endif

   