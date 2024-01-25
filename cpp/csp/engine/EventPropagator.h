#ifndef _IN_CSP_ENGINE_EVENTPROPAGATOR_H
#define _IN_CSP_ENGINE_EVENTPROPAGATOR_H

#include <csp/engine/InputId.h>
#include <stdint.h>

namespace csp
{

class Consumer;

class EventPropagator
{
public:
    EventPropagator();

    bool addConsumer( Consumer * consumer, InputId id, bool checkExists = false )
    {
        return m_consumers.addConsumer( consumer, id, checkExists );
    }

    bool removeConsumer( Consumer * consumer, InputId id )
    {
        return m_consumers.removeConsumer( consumer, id );
    }

    void clear() { m_consumers.clear(); }

    void propagate();

    template<typename CB >
    void apply( CB && cb ) const
    {
        m_consumers.apply( std::forward<CB>( cb ) );
    }

private:
    //Brief explanation on what we're doing here.  We're trying to optimize a few things
    // -skip iteration over consumers in the common case where there is only 1
    // -minimize size of the object.  avoid using std::vector ( 24 bytes ) since we can do with 32 bit sizes, so ConsumerVector is 16 bytes
    // Below we store a union of a single ConsumerInfo (16 bytes) and a ConsumerVector ( also 16 bytes )
    // We have to distinguish the state of the union as either EMPTY, SINGLE consumer or VECTOR.  We do this by flagging the LSB of the first 
    // first element of ConsumerInfo | ConsumerVector, which both happen to be a pointer type.  We init the flag to EMPTY ( 0x2 ).
    // if we add an elemnt the flag is cleared, this denotes we're in SINGLE mode ( single's ConsumerInfo -> consumer access does not need to unmask )
    // if we are in VECTOR mode, the flag is set to 0x1.  ConsumerVector has to unmask all accesses to its ConsumerInfo * pointer.
    //
    // To summarize:
    //     flag has 0x2 set - EMPTY
    //     flag has 0x1 set - VECTOR
    //     flag is not set  - SINGLE
    struct ConsumerInfo
    {
        ConsumerInfo() : consumer ( nullptr ), inputId( -1 ) {}

        ConsumerInfo( Consumer * c, InputId i ) : consumer( c ), inputId( i )
        {}

        bool operator==( const ConsumerInfo & rhs ) const { return rhs.consumer == consumer && rhs.inputId == inputId; }

        Consumer * consumer;
        InputId    inputId;
    };
    
    class ConsumerVector
    {
    public:
        ConsumerVector();
        ~ConsumerVector();

        ConsumerVector( const ConsumerVector & ) = delete;
        ConsumerVector & operator=( const ConsumerVector & ) = delete;

        bool addConsumer( Consumer * consumer, InputId id, bool checkExists );
        bool removeConsumer( Consumer * consumer, InputId id );

        ConsumerInfo * consumers()
        { 
            //need to unmask the flag bit
            return ( ConsumerInfo * ) ( ( ( uint64_t ) m_consumers ) & ~1 );
        }

        const ConsumerInfo * consumers() const { return const_cast<ConsumerVector *>( this ) -> consumers(); }


        uint32_t       size() const { return m_size; }
        void           clear()      { m_size = 0; }

        ConsumerInfo & operator[]( size_t i )             { return consumers()[i]; }
        const ConsumerInfo & operator[]( size_t i ) const { return consumers()[i]; }

    private:

        void push_back( Consumer * consumer, InputId id );
        ConsumerInfo * findConsumer( Consumer * c, InputId i );

        //we're sacrificing speed of removals for memory over here, make active/make passive will do a linear scan
        //the expectation is that those calls should be relatively rare during runtime
        ConsumerInfo * m_consumers = nullptr;
        uint32_t       m_size = 0;
        uint32_t       m_capacity = 0;
    };

    class Consumers
    {
    public:
        Consumers();
        ~Consumers();

        bool addConsumer( Consumer * consumer, InputId id, bool checkExists );
        bool removeConsumer( Consumer * consumer, InputId id );
        void clear();

        template<typename CB >
        void apply( CB && cb ) const
        {
            if( likely( !isEmpty() ) )
            {
                if( isSingle() )
                    cb( single().consumer, single().inputId );
                else
                {
                    const ConsumerInfo * ci  = consumersVector().consumers();
                    const ConsumerInfo * end = ci + consumersVector().size();
                    while( ci < end )
                    {
                        cb( ci -> consumer, ci -> inputId );
                        ++ci;
                    }
                }
            }
        }

    private:
        //sentinel to distinguish completely empty set from unset single consumer
        static inline void * EMPTY = reinterpret_cast<void *>( 0x2 );

        bool isEmpty() const               { return m_consumerData.flag == EMPTY; }
        bool isVector() const              { return uint64_t( m_consumerData.flag ) & 0x1; }
        bool isSingle() const              { return !isVector(); }

        ConsumerVector & consumersVector() { return m_consumerData.consumers; }
        ConsumerInfo & single()            { return m_consumerData.singleConsumer; }

        const ConsumerVector & consumersVector() const { return m_consumerData.consumers; }
        const ConsumerInfo & single() const            { return m_consumerData.singleConsumer; }

        union ConsumerData
        {
            // both object below start with a pointer type.  We flag the last bit of the pointer if its a vector
            // (when vector accesses its ptr, it has to mask out, single mode does not have to since we know its not set)
            void * flag;
            ConsumerInfo   singleConsumer;
            ConsumerVector consumers;

            ConsumerData() {};
            ~ConsumerData() {};

        } m_consumerData;
    };

    Consumers m_consumers;
};

};
#endif
