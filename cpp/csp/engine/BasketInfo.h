#ifndef _IN_CSP_ENGINE_BASKETINFO_H
#define _IN_CSP_ENGINE_BASKETINFO_H

#include <csp/core/Exception.h>
#include <csp/engine/InputId.h>
#include <csp/engine/RootEngine.h>
#include <csp/engine/TimeSeriesProvider.h>
#include <vector>

namespace csp
{

class Node;

class InputBasketInfo
{
    using TickedInputs = std::vector<INOUT_ELEMID_TYPE>;

public:
    InputBasketInfo( RootEngine * rootEngine, size_t size, bool isDynamic = false );
    ~InputBasketInfo();

    InputBasketInfo( const InputBasketInfo & ) = delete;
    InputBasketInfo & operator=( const InputBasketInfo & ) = delete;

    bool isDynamicBasket() const { return m_isDynamic; }

    const TimeSeriesProvider * elem( int64_t elemId ) const
    {
        CSP_ASSERT( ( elemId >= 0 && elemId < m_size ) || ( elemId == -1 && m_isDynamic ) );
        return m_inputs[ elemId ];
    }

    void setElem( int64_t elemId, const TimeSeriesProvider * ts )
    {
        CSP_ASSERT( !m_inputs[ elemId ] );
        m_inputs[ elemId ] = ts;
    }

    size_t size() const { return m_size; }

    void handleEvent( int64_t elemId )
    {
        const auto * ts = elem( elemId );
        if( ts -> lastCycleCount() > m_lastCycleCount )
        {
            m_tickedInputs.clear();
            m_lastCycleCount = ts -> lastCycleCount();
        }

        m_tickedInputs.emplace_back( elemId );
    }

    bool allValid() const
    {
        if( likely( m_valid ) )
            return m_valid;

        //Unforunately we have to loop over every input if we're not valid yet.  Originally we were counting
        //valid in handleEvent but this breaks if the basket is passive
        for( auto it = m_inputs, itEnd = m_inputs + m_size; it != itEnd; ++it )
        {
            if( !(*it) -> valid() )
                return false;
        }
        m_valid = true;
        return true;
    }

    //num ticked this cycle
    bool ticked() const      { return hasTickedInputs(); }

    struct input_iterator
    {
        input_iterator() : m_it( nullptr ), m_itEnd( nullptr ), m_elemId(0) {}

        input_iterator( TimeSeriesProvider const * const * itBegin, TimeSeriesProvider const * const * itEnd,
                        int64_t startId = 0 ) : m_elemId( startId )
        {
            m_it    = itBegin;
            m_itEnd = itEnd;
        }

        const TimeSeriesProvider * get() const         { return *m_it; }
        const TimeSeriesProvider * operator ->() const { return get(); }
        const TimeSeriesProvider & ts() const          { return *get(); }

        operator bool() const { return m_it != m_itEnd; }

        input_iterator & operator++() { ++m_it; ++m_elemId; return *this; }
        int64_t elemId() const        { return m_elemId; }

    private:
        TimeSeriesProvider const * const * m_it;
        TimeSeriesProvider const * const * m_itEnd;
        int64_t m_elemId;
    };

    struct ticked_iterator
    {
        ticked_iterator( TimeSeriesProvider const * const * inputs,
                         TickedInputs::const_iterator it,
                         TickedInputs::const_iterator itEnd ) : m_inputs( inputs ),
                                                                m_it( it ),
                                                                m_itEnd( itEnd )
        {}

        operator bool() const          { return m_it != m_itEnd; }
        ticked_iterator & operator++() { ++m_it; return *this; }

        const TimeSeriesProvider * get() const          { return m_inputs[ *m_it ]; }
        const TimeSeriesProvider * operator -> () const { return get(); }
        const TimeSeriesProvider & ts() const           { return *get(); }

        int64_t elemId() const                          { return *m_it; }

    private:
        TimeSeriesProvider const * const * m_inputs;
        TickedInputs::const_iterator m_it;
        TickedInputs::const_iterator m_itEnd;
    };

    struct valid_iterator
    {
        valid_iterator( TimeSeriesProvider const * const * itBegin,
                        TimeSeriesProvider const * const * itEnd ) : m_it( itBegin ),
                                                                     m_itEnd( itEnd ),
                                                                     m_elemId( 0 )
        {
            if( m_it != m_itEnd && unlikely( !(*m_it ) -> valid() ) )
                ++(*this);
        }

        operator bool() const { return m_it != m_itEnd; }

        valid_iterator & operator++()
        {
            do
            {
                ++m_it;
                ++m_elemId;
            } while( m_it != m_itEnd && !( (*m_it) -> valid() ) );
            return *this;
        }

        int64_t elemId() const                          { return m_elemId; }
        const TimeSeriesProvider * get() const          { return *m_it; }
        const TimeSeriesProvider * operator -> () const { return get(); }
        const TimeSeriesProvider & ts() const           { return *get(); }

    private:
        TimeSeriesProvider const * const * m_it;
        TimeSeriesProvider const * const * m_itEnd;
        int64_t                            m_elemId;
    };

    input_iterator  begin_inputs() const { return input_iterator( m_inputs, m_inputs + m_size ); }
    ticked_iterator begin_ticked() const
    {
        if( hasTickedInputs() )
            return ticked_iterator( m_inputs, m_tickedInputs.begin(), m_tickedInputs.end() );

        return ticked_iterator( m_inputs, m_tickedInputs.end(), m_tickedInputs.end() );
    }

    valid_iterator  begin_valid() const  { return valid_iterator( m_inputs, m_inputs + m_size ); }

    input_iterator  begin_inputs( bool include_hidden ) const
    {
        if( include_hidden && m_isDynamic )
            return input_iterator( m_inputs - 1, m_inputs + m_size, -1 );
        return input_iterator( m_inputs, m_inputs + m_size );
    }

protected:
    bool hasTickedInputs() const
    {
        return m_lastCycleCount == m_rootEngine -> cycleCount();
    }

    //not owned
    TimeSeriesProvider const ** m_inputs;
    TickedInputs                m_tickedInputs;
    INOUT_ELEMID_TYPE           m_size;
    uint64_t                    m_lastCycleCount;
    RootEngine *                m_rootEngine;
    mutable bool                m_valid;
    bool                        m_isDynamic;
};

class DynamicInputBasketInfo : public InputBasketInfo
{
public:
    using ChangeCallback = std::function<void(const DialectGenericType & key,bool added, int64_t elemId, int64_t replaceId )>;
    using Keys = std::vector<DialectGenericType>;

    DynamicInputBasketInfo( RootEngine * rootEngine ) : InputBasketInfo( rootEngine, 0, true ),
                                                        m_capacity(0), m_tickCountPolicy( 1 )
    {
        //Note that dynamic baskets are always valid, no need to compute it
        m_valid = true;
    }

    ~DynamicInputBasketInfo() {}

    DynamicInputBasketInfo( const DynamicInputBasketInfo & ) = delete;
    DynamicInputBasketInfo & operator=( const DynamicInputBasketInfo & ) = delete;


    void setChangeCallback( ChangeCallback cb ) { m_changeCallback = std::move( cb ); }

    const TimeSeriesProvider * shapeTs() const { return m_inputs[-1]; }

    int64_t addDynamicKey( const DialectGenericType & key, const TimeSeriesProvider * ts );

    //removeID is element to remove, replaceId ( if >= 0 ) is element to take its place ( compaction )
    void removeDynamicKey( uint64_t engineCycle, const DialectGenericType & key, int64_t removeId, int64_t replaceId );

    //for dynamic input baskets we maintain bufffering policy on the dynamic basket so that they get applied to newly added inputs
    void setTickCountPolicy( int32_t tickCount )     { m_tickCountPolicy = tickCount; }
    void setTickTimeWindowPolicy( TimeDelta window ) { m_timeWindowPolicy = window; }

private:
    ChangeCallback    m_changeCallback;
    INOUT_ELEMID_TYPE m_capacity;
    int32_t           m_tickCountPolicy;
    TimeDelta         m_timeWindowPolicy;
};

class OutputBasketInfo
{
public:
    OutputBasketInfo( CspTypePtr & type, Node * node, size_t size, bool isDynamic = false );
    ~OutputBasketInfo();

    OutputBasketInfo( const OutputBasketInfo & rhs ) = delete;
    OutputBasketInfo & operator=( const OutputBasketInfo & rhs ) = delete;

    TimeSeriesProvider * elem( int64_t elemId ) const
    {
        CSP_ASSERT( elemId >= 0 && elemId < m_size );
        return m_outputs[ elemId ];
    }

    size_t size() const { return m_size; }

    bool isDynamicBasket() const { return m_isDynamic; }

protected:

    //owned
    TimeSeriesProvider  ** m_outputs;
    INOUT_ELEMID_TYPE      m_size;
    bool                   m_isDynamic;
};

class DynamicOutputBasketInfo : public OutputBasketInfo
{
public:
    DynamicOutputBasketInfo( CspTypePtr & type, Node * node );
    ~DynamicOutputBasketInfo();

    DynamicOutputBasketInfo( const DynamicOutputBasketInfo & rhs ) = delete;
    DynamicOutputBasketInfo & operator=( const DynamicOutputBasketInfo & rhs ) = delete;

    //returns element id of added key, pushes shape tick
    int64_t addDynamicKey( const DialectGenericType & key );

    //this is used exclusively by DynamicNode ( dynamic graphs ).  See comment below on m_ownTs;
    int64_t addDynamicTs( const DialectGenericType & key, const TimeSeriesProvider * ts );

    //when we remove elem N, we take elem at the end of the basket and move it into its slot
    //returns index of ts moved from end to take this slot
    int64_t removeDynamicKey( const DialectGenericType & key, int64_t elemId );

    void linkInputBasket( Node * node, INOUT_ID_TYPE inputIdx );

private:
    //called for every add
    bool addCapacity();
    void propagateAddKey( const DialectGenericType & key, const TimeSeriesProvider * ts );
    void addShapeChange( const DialectGenericType & key, bool added );

    TimeSeriesProvider m_shapeTs;
    CspTypePtr         m_elemType;
    Node             * m_parentNode;
    INOUT_ELEMID_TYPE  m_capacity;

    //for dynamic graph basket outputs we take an existing ts and stick it in the dynamic output basket
    //for efficiency ( alternative is to own a copy on the outpu basket and copy every tick, which we want to avoid )
    bool               m_ownTs;
};

}

#endif
