#include <csp/core/Exception.h>
#include <csp/engine/Consumer.h>
#include <csp/engine/CycleStepTable.h>
#include <csp/engine/Profiler.h>
#include <string.h>

namespace csp
{

static Consumer *s_END_MARKER = reinterpret_cast<Consumer*>( 0x1 );

CycleStepTable::CycleStepTable() : m_maxRank( -1 ), m_rankBitset()
{
}

CycleStepTable::~CycleStepTable()
{
}

void CycleStepTable::resize( int32_t maxRank )
{
    if( maxRank > m_maxRank )
    {
        m_maxRank = maxRank;
        m_table.resize( m_maxRank + 1, { nullptr, nullptr } );
        m_rankBitset.resize( m_maxRank + 1 );
    }
}

void CycleStepTable::schedule( Consumer * node )
{
    //already scheduled
    if( node -> next() != nullptr )
        return;

    int32_t rank = node -> rank();
    auto & entry = m_table[ rank ];

    if( !entry.head )
    {
        m_rankBitset.set( rank );
        entry.head = entry.tail = node;
    }
    else
    {
        entry.tail -> setNext( node );
        entry.tail = node;
    }

    node -> setNext( s_END_MARKER );
}

void CycleStepTable::executeCycle( csp::Profiler * profiler, bool isDynamic )
{
    if( unlikely( profiler && !isDynamic ) ) // prioritize case without profiler
        profiler -> startCycle();

    auto curRank = m_rankBitset.find_first();
    while( curRank != DynamicBitSet<>::npos )
    {
        m_rankBitset.reset( curRank );
        auto * curConsumer = m_table[ curRank ].head;
        // no real need to set tail to nullptr
        m_table[ curRank ].head = nullptr;

        CSP_ASSERT( curConsumer != s_END_MARKER );
        
        while( curConsumer != s_END_MARKER )
        {
            if( unlikely( ( bool )profiler ) )
            {
                profiler -> startNode();
                curConsumer -> execute();
                profiler -> finishNode( curConsumer -> name() );
            }
            else
                curConsumer -> execute();

            Consumer * prevConsumer = curConsumer;
            curConsumer = curConsumer -> next();
            prevConsumer -> setNext( nullptr );
        }
        curRank = m_rankBitset.find_next( curRank );
    }

    if( unlikely( profiler && !isDynamic ) )
        profiler -> finishCycle();
}

}
