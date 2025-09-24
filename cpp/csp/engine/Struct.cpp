#include <csp/core/System.h>
#include <csp/engine/Struct.h>
#include <algorithm>

namespace csp
{

StructField::StructField( CspTypePtr type, const std::string & fieldname, 
                          size_t size, size_t alignment ) :
    m_fieldname( fieldname ),
    m_offset( 0 ),
    m_size( size ),
    m_alignment( alignment ),
    m_maskOffset( 0 ),
    m_maskBit( 0 ),
    m_maskBitMask( 0 ),
    m_type( type )
{
}

/*  StructMeta  

A note on member layout.  Meta will order objects in the following order:
- non-native fields ( ie PyObjectPtr for the python dialect )
- native fields sorted in order
- set/unset bitmask bytes ( 1 byte per 8 fields )

Derived structs will simply append to the layout of the base struct, properly padding between classes to align
its fields properly.  
This layout is imposed on Struct instances.  Since Struct needs refcount and meta * fields, for convenience they are stored
*before* a Struct's "this" pointer as hidden data.  This way struct ptrs can be passed into StructMeta without
and adjustments required for the hidden fields

*/

StructMeta::StructMeta( const std::string & name, const Fields & fields,
                        std::shared_ptr<StructMeta> base ) : m_name( name ), m_base( base ), m_fields( fields ),
                                                             m_size( 0 ), m_partialSize( 0 ), m_partialStart( 0 ), m_nativeStart( 0 ), m_basePadding( 0 ),
                                                             m_maskLoc( 0 ), m_maskSize( 0 ), m_firstPartialField( 0 ), m_firstNativePartialField( 0 ),
                                                             m_isPartialNative( true ), m_isFullyNative( true )
{
    if( m_fields.empty() && !m_base)
        CSP_THROW( TypeError, "Struct types must define at least 1 field" );

    //sort by sizes, biggest first, to get proper alignment
    //group generic objects separately at the start so that we can safely memcpy native types
    //decided to place them at the start cause they are most likely size of ptr or greater

    m_fieldnames.reserve( m_fields.size() );
    for( size_t i = 0; i < m_fields.size(); i++ ) 
        m_fieldnames.emplace_back( m_fields[i] -> fieldname() );

    std::sort( m_fields.begin(), m_fields.end(), []( auto && a, auto && b )
               {
                   return a -> isNative() < b -> isNative() ||
                       a -> size() > b -> size();
               } );

    size_t baseSize = m_base ? m_base -> size() : 0;
    size_t offset = baseSize;
    m_basePadding = 0;

    //align to first field's alignment
    if( m_fields.size() && ( offset % m_fields[0] -> alignment() != 0 ) )
        m_basePadding = m_fields[0] -> alignment() - offset % m_fields[0] -> alignment();

    offset += m_basePadding;
    if( !m_fields.empty() )
        CSP_ASSERT( ( offset % m_fields[0] -> alignment() ) == 0 );

    m_partialStart = offset;
    m_nativeStart  = m_partialStart;

    for( size_t idx = 0; idx < m_fields.size(); ++idx )
    {
        auto & f = m_fields[ idx ];
        if( offset % f -> alignment() != 0 )
            offset += f -> alignment() - offset % f -> alignment();

        f -> setOffset( offset );

        offset        += f -> size();

        m_isPartialNative &= f -> isNative();

        if( !f -> isNative() )
        {
            m_nativeStart = offset;
            m_firstNativePartialField = idx + 1;
        }
    }

    m_isFullyNative = m_isPartialNative && ( m_base ? m_base -> isNative() : true );

    //Setup masking bits for our fields
    //NOTE we can be more efficient by sticking masks into any potential alignment gaps, dont want to spend time on it 
    //at this point
    m_maskSize     = !m_fields.empty() ? 1 + ( ( m_fields.size() - 1 ) / 8 ) : 0;
    m_size         = offset + m_maskSize;
    m_partialSize  = m_size - baseSize;
    m_maskLoc      = m_size - m_maskSize;

    size_t  maskLoc = m_maskLoc;
    uint8_t maskBit = 0;
    for( auto & f : m_fields )
    {
        f -> setMaskOffset( maskLoc, maskBit );
        if( ++maskBit == 8 )
        {
            maskBit = 0;
            ++maskLoc;
        }
    }

    if( m_base )
    {
        m_fields.insert( m_fields.begin(), m_base -> m_fields.begin(), m_base -> m_fields.end() );
        m_fieldnames.insert( m_fieldnames.begin(), m_base -> m_fieldnames.begin(), m_base -> m_fieldnames.end() );
        
        m_firstPartialField        = m_base -> m_fields.size();
        m_firstNativePartialField += m_base -> m_fields.size();
        m_fieldMap = m_base -> m_fieldMap;
    }

    for( size_t idx = m_firstPartialField; idx < m_fields.size(); ++idx )
    {
        auto rv = m_fieldMap.emplace( m_fields[ idx ] -> fieldname().c_str(), m_fields[ idx ] );
        if( !rv.second )
            CSP_THROW( ValueError, "csp Struct " << name << " attempted to add existing field " << m_fields[ idx ] -> fieldname() );
    }
}

StructMeta::~StructMeta()
{
    m_default.reset();
}

Struct * StructMeta::createRaw() const
{
    Struct * s;
    //This weird looking new call actually is calling Struct::opereatore new( size_t, std::shared_ptr<StructMeta> )
    //its not actually doing an in-place on StructMeta this
    s = new ( shared_from_this() ) Struct( shared_from_this() );

    initialize( s );

    if( m_default )
        s -> deepcopyFrom( m_default.get() );

    return s;
}

std::string StructMeta::layout() const
{
    std::string out;
    out.resize( size(), ' ' );

    for( auto & field : m_fields )
    {
        char type = ' ';
        switch( field -> type() -> type() )
        {
            case CspType::Type::BOOL:             type = 'b'; break;
            case CspType::Type::INT8:             type = 'c'; break;
            case CspType::Type::UINT8:            type = 'C'; break;
            case CspType::Type::INT16:            type = 'h'; break;
            case CspType::Type::UINT16:           type = 'H'; break;
            case CspType::Type::INT32:            type = 'd'; break;
            case CspType::Type::UINT32:           type = 'D'; break;
            case CspType::Type::INT64:            type = 'l'; break;
            case CspType::Type::UINT64:           type = 'L'; break;
            case CspType::Type::DOUBLE:           type = 'f'; break;
            case CspType::Type::DATETIME:         type = 't'; break;
            case CspType::Type::TIMEDELTA:        type = 'e'; break;
            case CspType::Type::DATE:             type = 'y'; break;
            case CspType::Type::TIME:             type = 'T'; break;
            case CspType::Type::ENUM:             type = 'N'; break;
            case CspType::Type::STRUCT:           type = 'S'; break;
            case CspType::Type::STRING:           type = 's'; break;
            case CspType::Type::DIALECT_GENERIC:  type = 'G'; break;
            case CspType::Type::ARRAY:            type = 'a'; break;
            case CspType::Type::UNKNOWN:
            case CspType::Type::NUM_TYPES:
                break;
        }

        for( size_t c = 0; c < field -> size(); ++c )
            out[ field -> offset() + c ] = type;

        out[ field -> maskOffset() ] = 'M';
    }

    return out;
}

bool StructMeta::isDerivedType( const StructMeta * derived, const StructMeta * base )
{
    const StructMeta * b = derived;
    while( b && b != base )
        b = b -> m_base.get();
    return b != nullptr;
}

const StructMeta * StructMeta::commonBase( const StructMeta * x, const StructMeta *  y )
{
    const StructMeta * m = x;
    while( m && m != y )
        m = m -> m_base.get();

    if( !m )
    {
        m = y;
        while( m && m != x )
            m = m -> m_base.get();
    }
    return m;

}

void StructMeta::initialize( Struct * s ) const
{
    //TODO optimize initialize to use default if availbel instead of constructing
    //and then replaceing with default
    if( isNative() )
    {
        memset( reinterpret_cast<std::byte*>(s), 0, size() );
        return;
    }

    memset( reinterpret_cast<std::byte*>(s) + m_nativeStart, 0, partialNativeSize() );
    
    if( !m_isPartialNative )
    {
        for( size_t idx = m_firstPartialField; idx < m_firstNativePartialField; ++idx )
        {
            auto * field = m_fields[ idx ].get();
            static_cast<NonNativeStructField*>( field ) -> initialize( s );
        }
    }
    
    if( m_base )
        m_base -> initialize( s );
}

void StructMeta::copyFrom( const Struct * src, Struct * dest )
{
    if( unlikely( src == dest ) )
        return;

    const StructMeta * meta = commonBase( src -> meta(), dest -> meta() );
    if( !meta )
    {
        CSP_THROW( TypeError, "Attempting to copy from struct type '" << src -> meta() -> name() << "' to struct type '" << dest -> meta() -> name() 
                   << "'. copy_from may only be used to copy from struct with a common base type" );
    }

    meta -> copyFromImpl( src, dest, false );
}

void StructMeta::deepcopyFrom( const Struct * src, Struct * dest )
{
    if( unlikely( src == dest ) )
        return;

    const StructMeta * meta = commonBase( src -> meta(), dest -> meta() );
    if( !meta )
    {
        CSP_THROW( TypeError, "Attempting to deepcopy from struct type '" << src -> meta() -> name() << "' to struct type '" << dest -> meta() -> name() 
                   << "'. deepcopy_from may only be used to copy from struct with a common base type" );
    }
    
    meta -> copyFromImpl( src, dest, true );
}   

void StructMeta::copyFromImpl( const Struct * src, Struct * dest, bool deepcopy ) const
{
    //quick outs, if fully native we can memcpy the whole thing
    if( isNative() )
        memcpy( reinterpret_cast<std::byte*>(dest), reinterpret_cast<const std::byte*>(src), size() );
    else
    {
        //check if we have non-native types to handle here
        if( !m_isPartialNative )
        {
            //this logic relies on us allocating non-native fields up front
            for( size_t idx = m_firstPartialField; idx < m_firstNativePartialField; ++idx )
            {
                auto * field = m_fields[ idx ].get();

                if( field -> isSet( src ) )
                    if( deepcopy )
                        static_cast<NonNativeStructField*>( field ) -> deepcopyFrom( src, dest );
                    else
                        static_cast<NonNativeStructField*>( field ) -> copyFrom( src, dest );
                else
                    static_cast<NonNativeStructField*>( field ) -> clearValue( dest );
            }
        }

        //note that partialNative will include the mask bytes - this sets the native part and the mask
        memcpy( reinterpret_cast<std::byte*>(dest) + m_nativeStart, reinterpret_cast<const std::byte*>(src) + m_nativeStart,
                partialNativeSize() );
     
        if( m_base )
            m_base -> copyFromImpl( src, dest, deepcopy );
    }
}

void StructMeta::updateFrom( const Struct * src, Struct * dest )
{
    if( unlikely( src == dest ) )
        return;

    const StructMeta * meta = commonBase( src -> meta(), dest -> meta() );
    
    if( !meta )
    {
        CSP_THROW( TypeError, "Attempting to update from struct type '" << src -> meta() -> name() << "' to struct type '" << dest -> meta() -> name() 
                   << "'. update_from may only be used to update from struct with a common base type" );
    }

    meta -> updateFromImpl( src, dest );
}    

void StructMeta::updateFromImpl( const Struct * src, Struct * dest ) const
{
    for( Fields::const_iterator it = m_fields.begin() + m_firstPartialField; it != m_fields.end(); ++it )
    {
        const StructFieldPtr & field = *it;
        if( field -> isSet( src ) )
            field -> copyFrom( src, dest );
    }

    // update doesn't unset fields, so set the mask to the bitwise or of src and dest masks
    for( size_t i = m_maskLoc; i < m_maskLoc + m_maskSize; i++ )
    {
        std::byte * destMaskByte_ptr = reinterpret_cast<std::byte*>(dest) + i;
        const std::byte * srcMaskByte_ptr = reinterpret_cast<const std::byte*>(src) + i;
        *destMaskByte_ptr = *destMaskByte_ptr | *srcMaskByte_ptr;
    }

    if( m_base )
        m_base -> updateFromImpl( src, dest );
}


bool StructMeta::isEqual( const Struct * x, const Struct * y ) const
{
    if( x -> meta() != y -> meta() )
        return false;

    //Note the curent use of memcpy for native types.  This can cause issues on double comparisons
    //esp if expecting NaN == NaN to be false, and when comparing -0.0 to +0.0.. may want to revisit
    //We we do we may as well remove the basepadding copy 
    if( isNative() )
        return memcmp( x, y, size() ) == 0;

    int rv = memcmp( x + m_nativeStart, y + m_nativeStart, partialNativeSize() );
    if( rv != 0 )
        return false;

    if( !m_isPartialNative )
    {
        for( size_t idx = m_firstPartialField; idx < m_firstNativePartialField; ++idx )
        {
            auto * field = m_fields[ idx ].get();
                
            if( field -> isSet( x ) != field -> isSet( y ) )
                return false;

            if( field -> isSet( x ) )
            {
                bool rv = static_cast<NonNativeStructField*>( field ) -> isEqual( x, y );
                if( !rv )
                    return false;
            }
        }
    }

    if( m_base )
        return m_base -> isEqual( x, y );

    return true;
}

template <typename T>
struct ScopedIncrement
{
    ScopedIncrement( T & v ) : m_v{ v } { ++m_v; }
    ~ScopedIncrement() { --m_v; }

private:
    ScopedIncrement() = delete;
    T & m_v;
};

size_t StructMeta::hash( const Struct * x ) const
{
    size_t hash = std::hash<uint64_t>()( (uint64_t ) x -> meta() );
    if( isNative() )
        return hash ^ csp::hash::hash_bytes( x, size() );

    static constexpr size_t MAX_RECURSION_DEPTH = 1000;
    static thread_local size_t s_recursionDepth = 0;
    ScopedIncrement guard( s_recursionDepth );

    if ( unlikely( s_recursionDepth > MAX_RECURSION_DEPTH ) )
        CSP_THROW( RecursionError,
            "Exceeded max recursion depth of " << MAX_RECURSION_DEPTH << " in " << name() << "::hash(), cannot hash cyclic data structure" );

    hash ^= csp::hash::hash_bytes( x + m_nativeStart, partialNativeSize() );
    
    if( !m_isPartialNative )
    {
        for( size_t idx = m_firstPartialField; idx < m_firstNativePartialField; ++idx )
        {
            auto * field = m_fields[ idx ].get();
            
            //we dont incorporate unset fields, bitmask will cover them
            if( field -> isSet( x ) )
                hash ^= static_cast<NonNativeStructField*>( field ) -> hash( x );
        }
    }
    
    if( m_base )
        hash ^= std::hash<uint64_t>()( (uint64_t ) m_base.get() ) ^ m_base -> hash( x );

    return hash;
}

void StructMeta::clear( Struct * s ) const
{
    if( isNative() )
    {
        memset( reinterpret_cast<std::byte*>(s), 0, size() );
        return;
    }

    memset( reinterpret_cast<std::byte*>(s) + m_nativeStart, 0, partialNativeSize() );
    
    if( !m_isPartialNative )
    {
        for( size_t idx = m_firstPartialField; idx < m_firstNativePartialField; ++idx )
        {
            auto * field = m_fields[ idx ].get();
            
            if( field -> isSet( s ) )
                static_cast<NonNativeStructField*>( field ) -> clearValue( s );
        }
    }
    
    if( m_base )
        m_base -> clear( s );
}

bool StructMeta::allFieldsSet( const Struct * s ) const
{
    size_t  numLocalFields = m_fields.size() - m_firstPartialField;
    uint8_t numRemainingBits = numLocalFields % 8;

    const uint8_t * m = reinterpret_cast<const uint8_t *>( s ) + m_maskLoc;
    const uint8_t * e = m + m_maskSize - bool( numRemainingBits );
    for( ; m < e; ++m )
    {
        if( *m != 0xFF )
            return false;
    }

    if( numRemainingBits > 0 )
    {
        uint8_t bitmask = ( 1u << numRemainingBits ) - 1;
        if( ( *m & bitmask ) != bitmask )
            return false;
    }

    return m_base ? m_base -> allFieldsSet( s ) : true;
}

void StructMeta::destroy( Struct * s ) const
{
    if( isNative() )
        return;

    if( !m_isPartialNative )
    {
        for( size_t idx = m_firstPartialField; idx < m_firstNativePartialField; ++idx )
        {
            auto * field = m_fields[ idx ].get();
            static_cast<NonNativeStructField*>( field ) -> destroy( s );
        }
    }
    
    if( m_base )
        m_base -> destroy( s );
}

Struct::Struct( const std::shared_ptr<const StructMeta> & meta )
{
    //Initialize meta shared_ptr
    new( hidden() ) HiddenData();

    hidden() -> refcount   = 1;
    hidden() -> meta       = meta;
    hidden() -> dialectPtr = nullptr;
}

void * Struct::operator new( std::size_t count, const std::shared_ptr<const StructMeta> & meta )
{
    //allocate meta -> size() ( data members ) + room for hidden data refcount and meta ptr
    //Note that we currently assume sizeof( HiddenData ) is 16.  We expect it to be a multiple of 8
    //so all further fields are proper aligned ( again assuming our max aligned field is 8 bytes )
    //so be careful changing this assert if for some reason in the future sizeof( HiddenData ) changes
    static_assert( sizeof( HiddenData ) % 8 == 0 );

    void * ptr = ::operator new( sizeof( HiddenData ) + meta -> size() );
    return reinterpret_cast<uint8_t *>( ptr ) + sizeof( HiddenData );
}

void Struct::operator delete( void * ptr )
{
    void * p = reinterpret_cast<uint8_t *>( ptr ) - sizeof( HiddenData );
    ::operator delete( p );
}

}
