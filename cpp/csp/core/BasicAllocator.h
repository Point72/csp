#ifndef _IN_CSP_CORE_BASIC_ALLOCATOR_H
#define _IN_CSP_CORE_BASIC_ALLOCATOR_H

#include <csp/core/Platform.h>
#include <stdint.h>
#include <stdlib.h>
#include <list>
#include <string>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace csp
{

// Extremely basic non-thread safe fixed-size allocator
class BasicAllocator
{
public:
    //elemsize is size of a single alloc, blockSize is number of elements to 
    //allocate per block
    BasicAllocator();
    BasicAllocator( size_t elemSize, size_t blockSize, bool useHugePage, bool grow );
    ~BasicAllocator();

    void   init( size_t elemSize, size_t blockSize, bool useHugePage, bool grow );
    void * allocate();
    void   free( void * ptr );

private:
    void allocBlock();

    struct ArenaInfo
    {
        void * buffer;
        size_t size;
        bool   mmap;
    };

    using Arenas = std::list<ArenaInfo>;

    Arenas m_arenas;
    bool   m_grow;
    bool   m_useHugePage;
    size_t m_blockSize;
    size_t m_elemSize;
    void * m_freeptr;
};

inline BasicAllocator::BasicAllocator()
{}

inline BasicAllocator::BasicAllocator( size_t elemSize, size_t blockSize, 
                                       bool useHugePage, bool grow )
{
    init( elemSize, blockSize, useHugePage, grow );
}

inline void BasicAllocator::init( size_t elemSize, size_t blockSize, 
                                  bool useHugePage, bool grow )
{
    m_grow        = grow;
    m_useHugePage = useHugePage;
    m_blockSize   = blockSize;
    m_freeptr     = nullptr;
    m_elemSize    = std::max( elemSize, sizeof( void * ) );
    allocBlock();
}

inline BasicAllocator::~BasicAllocator()
{
    for( auto & entry : m_arenas )
    {
#ifdef __linux__
        if( entry.mmap )
            munmap( entry.buffer, entry.size );
        else
#endif
            ::free( entry.buffer );
    }       
}

inline void BasicAllocator::allocBlock()
{
    //keep doubleing size in new arenas
    size_t allocSize;
    size_t rawsize = allocSize = m_arenas.size() ? m_arenas.back().size * 2 : m_blockSize * m_elemSize;
    void * buffer;

#ifdef __linux__
    // NOTE: Only available on linux
    if( m_useHugePage )
    {
        //round to up to 2mb blocks for hugepage
        constexpr size_t _2mb = 2ul * 1024 * 1024;
        size_t hugesz = ( ( rawsize - 1 + _2mb ) / _2mb ) * _2mb;

        buffer = mmap( NULL, hugesz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB, -1, 0 );
        if( buffer == ( void * ) -1 )
        {
            buffer = malloc( rawsize );
            m_arenas.push_back( ArenaInfo{ buffer, rawsize, false } );;
        }
        else
        {
            m_arenas.push_back( ArenaInfo{ buffer, hugesz, true } );
            allocSize = hugesz;
        }
    }
    else
    {
        buffer = malloc( rawsize );
        m_arenas.push_back( ArenaInfo{ buffer, rawsize, false } );
    }
#else
    buffer = malloc( rawsize );
    m_arenas.push_back( ArenaInfo{ buffer, rawsize, false } );
#endif

    //link blocks
    size_t numBlocks = allocSize / m_elemSize;
    void * ptr = buffer;
    for( size_t i = 0; i < numBlocks - 1; ++i )
    {
        void * next = ( ( uint8_t * ) ptr ) + m_elemSize;
        *( uintptr_t ** )ptr = ( uintptr_t * ) next;
        ptr = next;
    }
    *( uintptr_t ** )ptr = ( uintptr_t * ) m_freeptr;

    m_freeptr = buffer;
}

inline void * BasicAllocator::allocate()
{
    if( m_freeptr != nullptr )
    {
        void * retval = m_freeptr;
        m_freeptr = *( uintptr_t ** ) m_freeptr;
        return retval;
    }

    if( m_grow )
    {
        allocBlock();
        return allocate();
    }

    return nullptr;
}

inline void BasicAllocator::free( void * ptr )
{
    void * prevhead = m_freeptr;
    m_freeptr = ptr;
    *( uintptr_t ** )m_freeptr = ( uintptr_t * ) prevhead;
}

template< typename T >
class TypedBasicAllocator : public BasicAllocator
{
public:
    TypedBasicAllocator() {}
    TypedBasicAllocator( size_t blockSize, bool useHugePage, bool grow ) { init( blockSize, useHugePage, grow ); }

    void init( size_t blockSize, bool useHugePage, bool grow ) { BasicAllocator::init( sizeof( T ), blockSize, useHugePage, grow ); }

    //note this does not construct
    T * allocate() { return static_cast<T *>( BasicAllocator::allocate() ); }
};

};

#endif
