#ifndef _IN_CSP_ENGINE_STRUCT_H
#define _IN_CSP_ENGINE_STRUCT_H

#include <csp/core/Hash.h>
#include <csp/engine/CspType.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace csp
{

class Struct;

template<typename T>
class TypedStructPtr;

using StructPtr = TypedStructPtr<Struct>;

class StructField
{
public:

    virtual ~StructField() {}

    const std::string & fieldname() const { return m_fieldname; }
    const CspTypePtr & type() const       { return m_type; }
    size_t  offset() const                { return m_offset; }      //offset to start of field's memory from start of struct mem
    size_t  size() const                  { return m_size; }        //size of field in bytes
    size_t  alignment() const             { return m_alignment; }   //alignment of the field
    size_t  maskOffset() const            { return m_maskOffset; }  //offset to location of the mask byte fo this field from start of struct mem
    uint8_t maskBit() const               { return m_maskBit; }     //bit within mask byte associated with this field
    uint8_t maskBitMask() const           { return m_maskBitMask; } //same as maskBit but as a mask ( 1 << bit

    bool isNative() const                 { return m_type -> type() <= CspType::Type::MAX_NATIVE_TYPE; }

    void setOffset( size_t off )          { m_offset = off; }
    void setMaskOffset( size_t off, uint8_t bit  )
    {
        CSP_ASSERT( bit < 8 );

        m_maskOffset  = off;
        m_maskBit     = bit;
        m_maskBitMask = 1 << bit;
    }

    bool isSet( const Struct * s ) const
    {
        const uint8_t * m = reinterpret_cast<const uint8_t *>( s ) + m_maskOffset;
        return (*m ) & m_maskBitMask;
    }

    //copy methods need not deal with mask set/unset, only copy values
    virtual void copyFrom( const Struct * src, Struct * dest ) const = 0;

    virtual void deepcopyFrom( const Struct * src, Struct * dest ) const = 0;

    template<typename T>
    struct upcast;

    //only use these if you know exactly what type you have!
    template<typename T>
    void setValue( Struct * s, const T & v ) const
    {
        static_cast<const typename upcast<T>::type *>( this ) -> setValue( s, v );
    }

    template<typename T>
    const T & value( const Struct * s ) const
    {
        return static_cast<const typename upcast<T>::type *>( this ) -> value( s );
    }

protected:

    StructField( CspTypePtr type, const std::string & fieldname,
                 size_t size, size_t alignment );

    void setIsSet( Struct * s ) const
    {
        uint8_t * m = reinterpret_cast<uint8_t *>( s ) + m_maskOffset;
        (*m) |= m_maskBitMask;
    }

    const void * valuePtr( const Struct * s ) const
    {
        return valuePtr( const_cast<Struct*>( s ) );
    }

    void * valuePtr( Struct * s ) const
    {
        return reinterpret_cast<uint8_t *>( s ) + m_offset;
    }

    void clearIsSet( Struct * s ) const
    {
        uint8_t * m = reinterpret_cast<uint8_t *>( s ) + m_maskOffset;
        (*m) &= ~m_maskBitMask;
    }

private:
    std::string  m_fieldname;
    size_t       m_offset;
    const size_t m_size;
    const size_t m_alignment;
    size_t       m_maskOffset;
    uint8_t      m_maskBit;
    uint8_t      m_maskBitMask;
    CspTypePtr   m_type;
};

using StructFieldPtr = std::shared_ptr<StructField>;

template<typename T>
class NativeStructField : public StructField
{
    static_assert( CspType::Type::fromCType<T>::type <= CspType::Type::MAX_NATIVE_TYPE );
    static_assert( sizeof(T) == alignof(T) );

public:
    NativeStructField() {}
    NativeStructField( const std::string & fieldname ) : NativeStructField( CspType::fromCType<T>::type(), fieldname )
    {
    }

    const T & value( const Struct * s ) const
    {
        return *reinterpret_cast<const T *>( valuePtr( s ) );
    }

    T & value( Struct * s ) const
    {
        return *reinterpret_cast<T*>( valuePtr( s ) );
    }

    void setValue( Struct * s, const T & v ) const
    {
        *reinterpret_cast<T*>( valuePtr( s ) ) = v;
        setIsSet( s );
    }

    void clearValue( Struct * s ) const
    {
        clearIsSet( s );
        memset( valuePtr( s ), 0, size() );
    }

    void copyFrom( const Struct * src, Struct * dest ) const override
    {
        value( dest ) = value( src );
    }

    void deepcopyFrom( const Struct * src, Struct * dest ) const override
    {
        value( dest ) = value( src );
    }

protected:
    NativeStructField( CspTypePtr type, const std::string & fieldname ) : StructField( type, fieldname, sizeof( T ), alignof( T ) )
    {}
};

using BoolStructField      = NativeStructField<bool>;
using Int8StructField      = NativeStructField<int8_t>;
using UInt8StructField     = NativeStructField<uint8_t>;
using Int16StructField     = NativeStructField<int16_t>;
using UInt16StructField    = NativeStructField<uint16_t>;
using Int32StructField     = NativeStructField<int32_t>;
using UInt32StructField    = NativeStructField<uint32_t>;
using Int64StructField     = NativeStructField<int64_t>;
using UInt64StructField    = NativeStructField<uint64_t>;
using DoubleStructField    = NativeStructField<double>;
using DateTimeStructField  = NativeStructField<DateTime>;
using TimeDeltaStructField = NativeStructField<TimeDelta>;
using DateStructField      = NativeStructField<Date>;
using TimeStructField      = NativeStructField<Time>;

class CspEnumStructField final : public NativeStructField<CspEnum>
{
public:
    CspEnumStructField( CspTypePtr type, const std::string & fieldname ) : NativeStructField( type, fieldname )
    {}
};

template<typename T>
class NotImplementedStructField : public StructField
{
public:
    const T & value( const Struct * s ) const
    {
        CSP_THROW( NotImplemented, "Struct fields are not supported for type " << CspType::Type::fromCType<T>::type );
    }

    void setValue( Struct * s, const T & v ) const
    {
        CSP_THROW( NotImplemented, "Struct fields are not supported for type " << CspType::Type::fromCType<T>::type );
    }

    void clearValue( Struct * s ) const
    {
        CSP_THROW( NotImplemented, "Struct fields are not supported for type " << CspType::Type::fromCType<T>::type );
    }

    void copyFrom( const Struct * src, Struct * dest ) const override
    {
        CSP_THROW( NotImplemented, "Struct fields are not supported for type " << CspType::Type::fromCType<T>::type );
    }

    void deepcopyFrom( const Struct * src, Struct * dest ) const override
    {
        CSP_THROW( NotImplemented, "Struct fields are not supported for type " << CspType::Type::fromCType<T>::type );
    }
};


//Non-native fields need to have these specialized in dialect-specific code
class NonNativeStructField : public StructField
{
public:
    NonNativeStructField( CspTypePtr type, const std::string &fieldname, size_t size, size_t alignment ) :
        StructField( type, fieldname, size, alignment )
    {}

    virtual void initialize( Struct * s ) const = 0;
    virtual void destroy( Struct * s ) const = 0;

    //methods can assume values are set, no need to check isSet
    virtual bool   isEqual( const Struct * x, const Struct * y ) const = 0;
    virtual size_t hash( const Struct * s ) const = 0;

    void clearValue( Struct * s ) const
    {
        clearValueImpl( s );
        clearIsSet( s );
    }
private:

    virtual void clearValueImpl( Struct * s ) const = 0;
};

class StringStructField final : public NonNativeStructField
{
public:
    using CType = csp::CspType::StringCType;

    StringStructField( CspTypePtr type, const std::string & fieldname ) :
        NonNativeStructField( type, fieldname, sizeof( CType ), alignof( CType ) )
    {}

    void initialize( Struct * s ) const override
    {
        new ( valuePtr( s ) ) CType();
    }

    void destroy( Struct *s ) const override
    {
        ( ( CType * ) valuePtr( s ) ) -> ~CType();
    }

    const CType & value( const Struct * s ) const
    {
        return value( const_cast<Struct *>( s ) );
    }

    void setValue( Struct * s, const char * str ) const
    {
        *reinterpret_cast<CType*>( valuePtr( s ) ) = str;
        setIsSet( s );
    }

    void setValue( Struct * s, const CType & str ) const
    {
        *reinterpret_cast<CType*>( valuePtr( s ) ) = str;
        setIsSet( s );
    }

    virtual void copyFrom( const Struct * src, Struct * dest ) const override
    {
        value( dest ) = value( src );
    }

    virtual void deepcopyFrom( const Struct * src, Struct * dest ) const override
    {
        value( dest ) = value( src );
    }

    virtual bool isEqual( const Struct * x, const Struct * y ) const override
    {
        return value( x ) == value( y );
    }

    virtual size_t hash( const Struct * x ) const override
    {
        return std::hash<CType>()( value( x ) );
    }

private:

    CType & value( Struct * s ) const
    {
        return *reinterpret_cast<CType *>( valuePtr( s ) );
    }

    void clearValueImpl( Struct * s ) const override
    {
        value( s ).clear();
    }
};

template<typename CType>
class ArrayStructField : public NonNativeStructField
{
    using ElemT = typename CType::value_type;

    template<typename T>
    static std::enable_if_t<CspType::isNative(CspType::Type::fromCType<T>::type), void> deepcopy( const std::vector<T> & src, std::vector<T> & dest )
    {
        dest = src;
    }
        
    static void deepcopy( const std::vector<std::string> & src, std::vector<std::string> & dest )
    {
        dest = src;
    }

    //Declared at end of file since StructPtr isnt defined yet
    static void deepcopy( const std::vector<StructPtr> & src, std::vector<StructPtr> & dest );

    static void deepcopy( const std::vector<DialectGenericType> & src, std::vector<DialectGenericType> & dest )
    {
        dest.resize( src.size() );
        for( size_t i = 0; i < src.size(); ++i )
            dest[i] = src[i].deepcopy();
    }

    template<typename StorageT>
    static void deepcopy( const std::vector<std::vector<StorageT>> & src, std::vector<std::vector<StorageT>> & dest )
    {
        dest.resize( src.size() );
        for( size_t i = 0; i < src.size(); ++i )
            deepcopy( src[i], dest[i] );
    }

public:
    ArrayStructField( CspTypePtr arrayType, const std::string & fieldname ) :
        NonNativeStructField( arrayType, fieldname, sizeof( CType ), alignof( CType ) )
    {}

    const CType & value( const Struct * s ) const
    {
        return value( const_cast<Struct *>( s ) );
    }

    void setValue( Struct * s, CType value ) const
    {
        *reinterpret_cast<CType*>( valuePtr( s ) ) = std::move( value );
        setIsSet( s );
    }

    void initialize( Struct * s ) const override
    {
        new ( valuePtr( s ) ) CType();
    }

    void destroy( Struct * s ) const override
    {
         ( ( CType * ) valuePtr( s ) ) -> ~CType();
    }

    void copyFrom( const Struct * src, Struct * dest ) const override
    {
        value( dest ) = value( src );
    }

    void deepcopyFrom( const Struct * src, Struct * dest ) const override
    {
        deepcopy( value( src ), value( dest ) );
    }
    
    bool isEqual( const Struct * x, const Struct * y ) const override
    {
        return value( x ) == value( y );
    }

    size_t hash( const Struct * s ) const override
    {
        return hash( value( s ) );
    }

private:

    template<typename V>
    size_t hash( const V & value ) const
    {
        static_assert(std::is_same<V,ElemT>::value || std::is_same<std::vector<V>,ElemT>::value );
        return std::hash<V>()( value );
    }

    template<typename V>
    size_t hash( const std::vector<V> & value ) const
    {
        size_t h = 1000003;

        for( auto const & v : value )
            h ^= hash( v );
        return h;
    }

    CType & value( Struct * s ) const
    {
        return *reinterpret_cast<CType *>( valuePtr( s ) );
    }

    void clearValueImpl( Struct * s ) const override
    {
        value(s).clear();
    }
};

class DialectGenericStructField : public NonNativeStructField
{
public:
    DialectGenericStructField( const std::string & fieldname, size_t size, size_t alignment ) :
        NonNativeStructField( CspType::DIALECT_GENERIC(), fieldname, size, alignment )
    {}

    const DialectGenericType & value( const Struct * s ) const
    {
        return value( const_cast<Struct *>( s ) );
    }

    DialectGenericType & value( Struct * s ) const
    {
        return *reinterpret_cast<DialectGenericType *>( valuePtr( s ) );
    }

    virtual void setValue( Struct * s, const DialectGenericType & obj ) const
    {
        value( s ) = obj;
        setIsSet( s );
    }

    void initialize( Struct * s ) const override
    {
        new( valuePtr( s ) ) DialectGenericType();
    }

    void destroy( Struct * s ) const override
    {
        ( ( DialectGenericType * ) valuePtr( s ) ) -> ~DialectGenericType();
    }

    void clearValueImpl( Struct * s ) const override
    {
        value( s ) = DialectGenericType();
    }

    void copyFrom( const Struct * src, Struct * dest ) const override
    {
        value( dest ) = value( src );
    }

    void deepcopyFrom( const Struct * src, Struct * dest ) const override
    {
        *( ( DialectGenericType * ) valuePtr( dest ) ) = ( ( DialectGenericType * ) valuePtr( src ) ) -> deepcopy();
    }

    bool isEqual( const Struct * x, const Struct * y ) const override
    {
        return value( x ) == value( y );
    }

    size_t hash( const Struct * x ) const override
    {
        return value(x).hash();
    }
};

template<typename T>
class TypedStructPtr
{
public:
    TypedStructPtr() : m_obj( nullptr ) {}
    ~TypedStructPtr()
    {
        if( m_obj )
            decref();
        m_obj = nullptr;
    }

    //takes ownership
    explicit TypedStructPtr( T * s ) : m_obj( s ) {}

    TypedStructPtr( const TypedStructPtr & rhs ) : m_obj( rhs.m_obj )
    {
        incref();
    }

    TypedStructPtr( TypedStructPtr && rhs ) : m_obj( rhs.m_obj )
    {
        rhs.m_obj = nullptr;
    }

    //typed ptr conversions
    template<typename U,typename = std::is_convertible<U*,T*>>
    TypedStructPtr( const TypedStructPtr<U> & rhs ) : m_obj( rhs.m_obj )
    {
        incref();
    }

    template<typename U,typename = std::is_convertible<U*,T*>>
    TypedStructPtr( TypedStructPtr<U> && rhs ) : m_obj( rhs.m_obj )
    {
        rhs.m_obj = nullptr;
    }

    TypedStructPtr & operator=( const TypedStructPtr & rhs )
    {
        if( m_obj )
            decref();
        m_obj = rhs.m_obj;
        if( m_obj )
            incref();
        return *this;
    }

    TypedStructPtr & operator=( TypedStructPtr && rhs )
    {
        if( m_obj )
            decref();
        m_obj = rhs.m_obj;
        rhs.m_obj = nullptr;
        return *this;
    }

    T * get()                     { return m_obj; }
    const T * get() const         { return m_obj; }

    T * operator ->()             { return m_obj; }
    const T * operator ->() const { return m_obj; }

    T & operator*()               { return *m_obj; }
    const T & operator*() const   { return *m_obj; }

    explicit operator bool() const { return m_obj != nullptr; }

    void reset()
    {
        if( m_obj )
            decref();
        m_obj = nullptr;
    }

    bool operator==( const TypedStructPtr<T> & rhs ) const;

    //StructPtr genericPtr() const { return structptr_cast<Struct>( *this ); }

private:

    //these are separate methods just for circular dep
    void decref();
    void incref();

    T * m_obj;

    template<typename U>
    friend class TypedStructPtr;

    template<typename V,typename U>
    friend TypedStructPtr<V> structptr_cast( const TypedStructPtr<U> & r );
};


template<typename T, typename U>
TypedStructPtr<T> structptr_cast( const TypedStructPtr<U> & r )
{
    TypedStructPtr<T> out( const_cast<T*>( static_cast<const T *>( r.get() ) ) );
    out.incref();
    return out;
}

class StructMeta : public std::enable_shared_from_this<StructMeta>
{
public:
    using Fields = std::vector<StructFieldPtr>;
    using FieldNames = std::vector<std::string>;

    //Fields will be re-arranged and assigned their offsets in StructMeta for optimal performance
    StructMeta( const std::string & name, const Fields & fields, std::shared_ptr<StructMeta> base = nullptr );
    virtual ~StructMeta();

    const std::string & name() const          { return m_name; }
    size_t size() const                       { return m_size; }
    size_t partialSize() const                { return m_partialSize; }

    bool isNative() const                     { return m_isFullyNative; }

    const Fields & fields() const             { return m_fields; }
    const FieldNames & fieldNames() const     { return m_fieldnames; }

    size_t maskLoc() const                    { return m_maskLoc; }
    size_t maskSize() const                   { return m_maskSize; }

    const StructFieldPtr & field( const char * name ) const
    {
        static StructFieldPtr s_empty;
        auto it = m_fieldMap.find( name );
        return it == m_fieldMap.end() ? s_empty : it -> second;
    }

    const StructFieldPtr & field( const std::string & name ) const
    {
        return field( name.c_str() );
    }

    void setDefault( StructPtr default_ ) { m_default = default_; }

    Struct *  createRaw() const;
    StructPtr create() const { return StructPtr( createRaw() ); }

    void   initialize( Struct * s ) const;
    void   destroy( Struct * s ) const;
    bool   isEqual( const Struct * x, const Struct * y ) const;
    size_t hash( const Struct * x ) const;
    static void copyFrom( const Struct * src, Struct * dest );
    static void deepcopyFrom( const Struct * src, Struct * dest );
    static void updateFrom( const Struct * src, Struct * dest );
    void   clear( Struct * s ) const;
    bool   allFieldsSet( const Struct * s ) const;

    //for debugging layout of types
    std::string layout() const;

    //returns true if derived == base or if derived is a derived type of base
    static bool isDerivedType( const StructMeta * derived, const StructMeta * base );
    //Given two types x and y this will return one or the other based on which is a base class of the other.
    //if x derives from y, returns y.  if y derives from x, returns x.  Otherwise this returns null.  NOTE this wont return a common ancestor
    static const StructMeta * commonBase( const StructMeta * x, const StructMeta * y );

    template<typename T>
    std::shared_ptr<typename StructField::upcast<T>::type> getMetaField( const char * fieldname, const char * expectedtype );

private:
    using FieldMap = std::unordered_map<const char *,StructFieldPtr, hash::CStrHash, hash::CStrEq >;

    size_t partialNativeSize()  const  { return m_size - m_nativeStart; }
    void   copyFromImpl( const Struct * src, Struct * dest, bool deepcopy ) const;
    void updateFromImpl( const Struct * src, Struct * dest ) const;

    std::string                 m_name;
    std::shared_ptr<StructMeta> m_base;
    StructPtr                   m_default;
    FieldMap                    m_fieldMap;

    //fields in order, memory owners of field objects which in turn own the key memory
    //m_fields includes all base fields as well. m_fieldnames maintains the proper iteration order of fields
    Fields                      m_fields;
    FieldNames                  m_fieldnames;
    size_t                      m_size;              // full size of struct including base classes - NOTE this does not include refcount and meta ptr on Struct
    size_t                      m_partialSize;       // size of data in this level of the hierarchy
    size_t                      m_partialStart;      // offset into mem where data for this level starts
    size_t                      m_nativeStart;       // offset into mem of where native members of this level start
    size_t                      m_basePadding;       // padded bytes between end of base to start of this level
    size_t                      m_maskLoc;           // mask location for this level
    size_t                      m_maskSize;          // number of bytes used for mask
    size_t                      m_firstPartialField; // index into m_fields of first field of this level
    size_t                      m_firstNativePartialField; // index into m_fields of first native field of this level
    bool                        m_isPartialNative;   // true if this level is all native
    bool                        m_isFullyNative;     // true if this level and all bases are fully native
};

template<typename T>
std::shared_ptr<typename StructField::upcast<T>::type> StructMeta::getMetaField( const char * fieldname, const char * expectedtype )
{
    auto field_ = field( fieldname );
    if( !field_ )
        CSP_THROW( TypeError, "Struct type " << name() << " missing required field " << fieldname << " for " << expectedtype );

    std::shared_ptr<typename StructField::upcast<T>::type> typedfield = std::dynamic_pointer_cast<typename StructField::upcast<T>::type>( field_ );
    if( !typedfield )
        CSP_THROW( TypeError, expectedtype << " - provided struct type " << name() << " expected type " << CspType::Type::fromCType<T>::type << " for field " << fieldname
                                           << " but got type " << field_ -> type() -> type() << " for " << expectedtype );

    return typedfield;
}

using StructMetaPtr = std::shared_ptr<StructMeta>;

class Struct
{
public:

    const StructMeta * meta() const { return hidden() -> meta.get(); }
    size_t refcount() const         { return hidden() -> refcount; }

    bool operator==( const Struct & rhs ) const { return meta() -> isEqual( this, &rhs ); }
    bool operator!=( const Struct & rhs ) const { return !( (*this) == rhs ); }

    size_t hash() const { return meta() -> hash( this ); }

    void clear()
    {
        meta() -> clear( this );
    }

    StructPtr copy() const
    {
        StructPtr copy = meta() -> create();
        copy -> copyFrom( this );
        return copy;
    }

    StructPtr deepcopy() const
    {
        StructPtr copy = meta() -> create();
        copy -> deepcopyFrom( this );
        return copy;
    }

    void copyFrom( const Struct * rhs )
    {
        StructMeta::copyFrom( rhs, this );
    }

    void deepcopyFrom( const Struct * rhs )
    {
        StructMeta::deepcopyFrom( rhs, this );
    }

    void updateFrom( const Struct * rhs )
    {
        StructMeta::updateFrom( rhs, this );
    }

    bool allFieldsSet() const
    {
        return meta() -> allFieldsSet( this );
    }


    //used to cache dialect representations of this struct, if needed
    void * dialectPtr() const      { return hidden() -> dialectPtr; }
    void setDialectPtr( void * p ) { hidden() -> dialectPtr = p; }

private:
    static void * operator new( std::size_t count, const std::shared_ptr<const StructMeta> & meta );
    static void operator delete( void * ptr, const std::shared_ptr<const StructMeta> & meta ) { delete( ( Struct * ) ptr ); }
    static void operator delete( void * ptr );

    Struct( const std::shared_ptr<const StructMeta> & meta );
    ~Struct()
    {
        meta() -> destroy( this );
    }

    void incref() { ++hidden() -> refcount; }

    void decref()
    {
//Work around GCC12 bug mis-identifying this code as use-after-free
#ifdef __linux__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuse-after-free"
#endif
        if( --hidden() -> refcount == 0 )
            delete this;
#ifdef __linux__
#pragma GCC diagnostic pop
#endif
    }


    template<typename T>
    friend class TypedStructPtr;
    friend class StructMeta;

    //Note these members are not included on size(), they're stored before "this" ptr ( see operator new / delete )
    struct HiddenData
    {
        size_t             refcount;
        std::shared_ptr<const StructMeta> meta;
        void             * dialectPtr;
    };

    const HiddenData * hidden() const
    {
        return const_cast<Struct *>( this ) -> hidden();
    }

    HiddenData * hidden()
    {
        return reinterpret_cast<HiddenData *>( reinterpret_cast<uint8_t *>( this ) - sizeof( HiddenData ) );
    }

    //actual data is allocated past this point
};

template<typename T>
inline void TypedStructPtr<T>::incref()
{
    m_obj -> incref();
}

template<typename T>
inline void TypedStructPtr<T>::decref()
{
    m_obj -> decref();
}


template<typename T>
bool TypedStructPtr<T>::operator==( const TypedStructPtr<T> & rhs ) const
{
    if( m_obj == nullptr || rhs.m_obj == nullptr )
        return m_obj == rhs.m_obj;

    return m_obj -> meta() -> isEqual( m_obj, rhs.m_obj );
}

//field that is another struct
class StructStructField final : public NonNativeStructField
{
public:
    StructStructField( CspTypePtr cspType, const std::string &fieldname ) :
        NonNativeStructField( cspType, fieldname, sizeof( StructPtr ), alignof( StructPtr ) )
    {
        CSP_ASSERT( cspType -> type() == CspType::Type::STRUCT );
        m_meta = std::static_pointer_cast<const CspStructType>( cspType ) -> meta();
    }

    const StructMetaPtr & meta() const { return m_meta; }

    void initialize( Struct * s ) const override
    {
        new ( valuePtr( s ) ) StructPtr();
    }

    void destroy( Struct * s ) const override
    {
        ( ( StructPtr * ) valuePtr( s ) ) -> ~StructPtr();
    }

    const StructPtr & value( const Struct * s ) const
    {
        return value( const_cast<Struct *>( s ) );
    }

    StructPtr & value( Struct * s ) const
    {
        return *reinterpret_cast<StructPtr *>( valuePtr( s ) );
    }

    void setValue( Struct * s, const StructPtr & obj ) const
    {
        value( s ) = obj;
        setIsSet( s );
    }

    virtual void copyFrom( const Struct * src, Struct * dest ) const override
    {
        value( dest ) = value( src );
    }

    virtual void deepcopyFrom( const Struct * src, Struct * dest ) const override
    {
        value( dest ) = value(src) -> deepcopy();
    }

    virtual bool isEqual( const Struct * x, const Struct * y ) const override
    {
        return ( *value( x ).get() ) == ( *value( y ).get() );
    }

    virtual size_t hash( const Struct * x ) const override
    {
        return value( x ) -> hash();
    }

private:
    void clearValueImpl( Struct * s ) const override
    {
        value( s ).reset();
    }

    StructMetaPtr m_meta;
};

//Defined here to break decl dep
template<typename ElemT>
void ArrayStructField<ElemT>::deepcopy( const std::vector<StructPtr> & src, std::vector<StructPtr> & dest )
{
    dest.resize( src.size() );
    for( size_t i = 0; i < src.size(); ++i )
        dest[i] = src[i] -> deepcopy();
}

template<typename T> struct StructField::upcast  { using type = NotImplementedStructField<T>; };

template<> struct StructField::upcast<bool>      { using type = BoolStructField; };
template<> struct StructField::upcast<int8_t>    { using type = Int8StructField; };
template<> struct StructField::upcast<uint8_t>   { using type = UInt8StructField; };
template<> struct StructField::upcast<int16_t>   { using type = Int16StructField; };
template<> struct StructField::upcast<uint16_t>  { using type = UInt16StructField; };
template<> struct StructField::upcast<int32_t>   { using type = Int32StructField; };
template<> struct StructField::upcast<uint32_t>  { using type = UInt32StructField; };
template<> struct StructField::upcast<int64_t>   { using type = Int64StructField; };
template<> struct StructField::upcast<uint64_t>  { using type = UInt64StructField; };
template<> struct StructField::upcast<double>    { using type = DoubleStructField; };
template<> struct StructField::upcast<DateTime>  { using type = DateTimeStructField; };
template<> struct StructField::upcast<TimeDelta> { using type = TimeDeltaStructField; };
template<> struct StructField::upcast<Date>      { using type = DateStructField; };
template<> struct StructField::upcast<Time>      { using type = TimeStructField; };
template<> struct StructField::upcast<CspEnum>   { using type = CspEnumStructField; };

template<> struct StructField::upcast<typename StringStructField::CType> { using type = StringStructField; };
template<> struct StructField::upcast<StructPtr>                         { using type = StructStructField; };
template<> struct StructField::upcast<csp::DialectGenericType>           { using type = DialectGenericStructField; };
template<typename StorageT> struct StructField::upcast<std::vector<StorageT>>
{ 
    static_assert( !std::is_same<StorageT,bool>::value, "vector<bool> should not be getting instantiated" );
    using type = ArrayStructField<std::vector<StorageT>>;
};

}

namespace std
{

template<>
struct hash<csp::StructPtr>
{
    size_t operator()( const csp::StructPtr & s ) const
    {
        return s -> hash();
    }
};

}

#endif
