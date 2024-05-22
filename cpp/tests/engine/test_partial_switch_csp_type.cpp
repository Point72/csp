#include <csp/core/Exception.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <gtest/gtest.h>
#include <iostream>
#include <csp/engine/TypeCast.h>

template< typename T, typename B >
auto getConverter( std::shared_ptr<const csp::CspType> curType )
{
    return csp::ConstructibleTypeSwitch<T>::invoke( curType . get(), []( auto tag )
    {
        return std::function( []( T val ) { return B(( typename decltype(tag)::type ) ( val )); } );
    } );
}

template< csp::CspType::TypeTraits::_enum v, typename B=long long, typename T=typename csp::CspType::Type::toCType<v>::type >
void runTypeConversionSupportTest( std::set<csp::CspType::Type> supportedTypes,
                                   std::set<csp::CspType::Type> unsupportedTypes )
{
    ASSERT_EQ( supportedTypes . size() + unsupportedTypes . size(), 10 );
    for( auto t:supportedTypes )
    {
        try
        {
            auto converter = getConverter<T, B>( std::make_shared<const csp::CspType>( t ));
            if( t != csp::CspType::Type::BOOL )
            {
                const bool specialIntCase = ( v == csp::CspType::Type::UINT64 ) && ( t == csp::CspType::Type::INT64 );
                if( !specialIntCase )
                {
                    ASSERT_EQ( B( std::numeric_limits<T>::max()), converter( std::numeric_limits<T>::max()));
                }
                ASSERT_EQ( std::numeric_limits<T>::min(), converter( std::numeric_limits<T>::min()));
            }
            else
            {
                ASSERT_EQ( std::numeric_limits<T>::max() != 0, converter( std::numeric_limits<T>::max()));
                ASSERT_EQ( std::numeric_limits<T>::min() != 0, converter( std::numeric_limits<T>::min()));
            }
        }
        catch( const csp::UnsupportedSwitchType &e )
        {
            throw;
        }
    }
    for( auto t:unsupportedTypes )
    {
        ASSERT_THROW(( getConverter<T, B>( std::make_shared<const csp::CspType>( t ))), csp::UnsupportedSwitchType );
    }
}


TEST( ConstructibleTypeSelectorTest, test_basic_type_support )
{
    runTypeConversionSupportTest<csp::CspType::Type::BOOL>( {
                                                                    csp::CspType::Type::BOOL,
                                                                    csp::CspType::Type::INT8,
                                                                    csp::CspType::Type::UINT8,
                                                                    csp::CspType::Type::INT16,
                                                                    csp::CspType::Type::UINT16,
                                                                    csp::CspType::Type::INT32,
                                                                    csp::CspType::Type::UINT32,
                                                                    csp::CspType::Type::INT64,
                                                                    csp::CspType::Type::UINT64,
                                                                    csp::CspType::Type::DOUBLE,
                                                            },
                                                            {} );

    runTypeConversionSupportTest<csp::CspType::Type::INT8>( {
                                                                    csp::CspType::Type::BOOL,
                                                                    csp::CspType::Type::INT8,
                                                                    csp::CspType::Type::INT16,
                                                                    csp::CspType::Type::INT32,
                                                                    csp::CspType::Type::INT64,
                                                                    csp::CspType::Type::DOUBLE },
                                                            { csp::CspType::Type::UINT8,
                                                              csp::CspType::Type::UINT16,
                                                              csp::CspType::Type::UINT32,
                                                              csp::CspType::Type::UINT64,
                                                            } );

    runTypeConversionSupportTest<csp::CspType::Type::UINT8>( {
                                                                     csp::CspType::Type::BOOL,
                                                                     csp::CspType::Type::UINT8,
                                                                     csp::CspType::Type::INT16,
                                                                     csp::CspType::Type::UINT16,
                                                                     csp::CspType::Type::INT32,
                                                                     csp::CspType::Type::UINT32,
                                                                     csp::CspType::Type::INT64,
                                                                     csp::CspType::Type::UINT64,
                                                                     csp::CspType::Type::DOUBLE },
                                                             {
                                                                     csp::CspType::Type::INT8
                                                             } );

    runTypeConversionSupportTest<csp::CspType::Type::INT16>( {
                                                                     csp::CspType::Type::BOOL,
                                                                     csp::CspType::Type::INT16,
                                                                     csp::CspType::Type::INT32,
                                                                     csp::CspType::Type::INT64,
                                                                     csp::CspType::Type::DOUBLE },
                                                             { csp::CspType::Type::UINT8,
                                                               csp::CspType::Type::INT8,
                                                               csp::CspType::Type::UINT16,
                                                               csp::CspType::Type::UINT32,
                                                               csp::CspType::Type::UINT64
                                                             } );

    runTypeConversionSupportTest<csp::CspType::Type::UINT16>( {
                                                                      csp::CspType::Type::BOOL,
                                                                      csp::CspType::Type::UINT16,
                                                                      csp::CspType::Type::INT32,
                                                                      csp::CspType::Type::UINT32,
                                                                      csp::CspType::Type::INT64,
                                                                      csp::CspType::Type::UINT64,
                                                                      csp::CspType::Type::DOUBLE },
                                                              {
                                                                      csp::CspType::Type::INT8,
                                                                      csp::CspType::Type::UINT8,
                                                                      csp::CspType::Type::INT16
                                                              } );

    runTypeConversionSupportTest<csp::CspType::Type::INT32>( {
                                                                     csp::CspType::Type::BOOL,
                                                                     csp::CspType::Type::INT32,
                                                                     csp::CspType::Type::INT64,
                                                                     csp::CspType::Type::DOUBLE },
                                                             { csp::CspType::Type::UINT8,
                                                               csp::CspType::Type::INT8,
                                                               csp::CspType::Type::INT16,
                                                               csp::CspType::Type::UINT16,
                                                               csp::CspType::Type::UINT32,
                                                               csp::CspType::Type::UINT64
                                                             } );

    runTypeConversionSupportTest<csp::CspType::Type::UINT32>( {
                                                                      csp::CspType::Type::BOOL,
                                                                      csp::CspType::Type::UINT32,
                                                                      csp::CspType::Type::INT64,
                                                                      csp::CspType::Type::UINT64,
                                                                      csp::CspType::Type::DOUBLE },
                                                              {
                                                                      csp::CspType::Type::INT8,
                                                                      csp::CspType::Type::UINT8,
                                                                      csp::CspType::Type::INT16,
                                                                      csp::CspType::Type::UINT16,
                                                                      csp::CspType::Type::INT32,
                                                              } );

    runTypeConversionSupportTest<csp::CspType::Type::INT64, double>( {
                                                                             csp::CspType::Type::BOOL,
                                                                             csp::CspType::Type::INT64,
                                                                             csp::CspType::Type::DOUBLE },
                                                                     { csp::CspType::Type::UINT8,
                                                                       csp::CspType::Type::INT8,
                                                                       csp::CspType::Type::INT16,
                                                                       csp::CspType::Type::UINT16,
                                                                       csp::CspType::Type::INT32,
                                                                       csp::CspType::Type::UINT32,
                                                                       csp::CspType::Type::UINT64
                                                                     } );

    runTypeConversionSupportTest<csp::CspType::Type::UINT64, double>( {
                                                                              csp::CspType::Type::BOOL,
                                                                              csp::CspType::Type::UINT64,
                                                                              csp::CspType::Type::INT64,
                                                                              csp::CspType::Type::DOUBLE },
                                                                      {
                                                                              csp::CspType::Type::INT8,
                                                                              csp::CspType::Type::UINT8,
                                                                              csp::CspType::Type::INT16,
                                                                              csp::CspType::Type::UINT16,
                                                                              csp::CspType::Type::INT32,
                                                                              csp::CspType::Type::UINT32,

                                                                      } );
    std::uint64_t aux = std::numeric_limits<const std::uint64_t>::max();
    ASSERT_THROW( csp::cast<int64_t>( std::numeric_limits<std::uint64_t>::max()), csp::RangeError );
    ASSERT_THROW( csp::cast<const int64_t>( std::numeric_limits<std::uint64_t>::max()), csp::RangeError );
    ASSERT_THROW( csp::cast<int64_t>( aux ), csp::RangeError );
    ASSERT_THROW( csp::cast<const int64_t>( aux ), csp::RangeError );

    runTypeConversionSupportTest<csp::CspType::Type::DOUBLE, double>( {
                                                                              csp::CspType::Type::BOOL,
                                                                              csp::CspType::Type::DOUBLE },
                                                                      {
                                                                              csp::CspType::Type::INT8,
                                                                              csp::CspType::Type::UINT8,
                                                                              csp::CspType::Type::INT16,
                                                                              csp::CspType::Type::UINT16,
                                                                              csp::CspType::Type::INT32,
                                                                              csp::CspType::Type::UINT32,
                                                                              csp::CspType::Type::INT64,
                                                                              csp::CspType::Type::UINT64,
                                                                      } );
}


template< typename T1, typename T2 >
void verifySameTypes()
{
    if( !std::is_same_v<T1, T2> )
    {
        static_assert( !std::is_same_v<T1, std::vector<void>> );
        static_assert( !std::is_same_v<T2, std::vector<void>> );
        CSP_THROW( csp::RuntimeException, "Types mismatch" << typeid( T1 ).name() << "," << typeid( T2 ).name());
    }
}

TEST( ArraySwitchTest, test_basic_switch )
{
    auto uint64Type      = csp::CspType::UINT64();
    auto uint64ArrayType = csp::CspArrayType::create( csp::CspType::UINT64());
    auto boolArrayType   = csp::CspArrayType::create( csp::CspType::BOOL());
    csp::PrimitiveCspTypeSwitch::invoke( uint64Type.get(), []( auto tag )
    {
        if( !std::is_same_v<typename decltype(tag)::type, std::uint64_t> )
        {
            CSP_THROW( csp::RuntimeException, "Dummy error" );
        }
    } );
    // Primitive shouldn't support arrays
    ASSERT_THROW( csp::PrimitiveCspTypeSwitch::invoke( uint64ArrayType.get(), []( auto tag ) {} ), csp::UnsupportedSwitchType );

    // Should work fine
    csp::PartialSwitchCspType<csp::CspType::Type::INT16, csp::CspType::Type::ARRAY>::invoke( uint64ArrayType.get(), []( auto tag )
    {
        verifySameTypes<typename decltype(tag)::type, std::vector<std::uint64_t>>();
    } );
    csp::PartialSwitchCspType<csp::CspType::Type::ARRAY>::invoke( uint64ArrayType.get(), []( auto tag )
    {
        verifySameTypes<typename decltype(tag)::type, std::vector<std::uint64_t>>();
    } );
    csp::PartialSwitchCspType<csp::CspType::Type::ARRAY>::invoke( boolArrayType.get(), []( auto tag )
    {
        verifySameTypes<typename decltype(tag)::type, std::vector<uint8_t>>();
    } );

    csp::PartialSwitchCspType<csp::CspType::Type::ARRAY>::invoke<csp::PartialSwitchCspType<csp::CspType::Type::BOOL>>(
            boolArrayType.get(), []( auto tag ) { verifySameTypes<typename decltype(tag)::type, std::vector<uint8_t>>(); } );
    csp::PartialSwitchCspType<csp::CspType::Type::ARRAY>::invoke<csp::PartialSwitchCspType<csp::CspType::Type::UINT64>>(
            uint64ArrayType.get(), []( auto tag ) { verifySameTypes<typename decltype(tag)::type, std::vector<std::uint64_t>>(); } );

    // Should raise since we support only array of bool
    ASSERT_THROW( csp::PartialSwitchCspType<csp::CspType::Type::ARRAY>::invoke<csp::PartialSwitchCspType<csp::CspType::Type::BOOL>>(
            uint64ArrayType.get(), []( auto tag ) {} ), csp::UnsupportedSwitchType );

    // Should raise since we support only array of uint64_t
    ASSERT_THROW( csp::PartialSwitchCspType<csp::CspType::Type::ARRAY>::invoke<csp::PartialSwitchCspType<csp::CspType::Type::UINT64>>(
            boolArrayType.get(), []( auto tag ) {} ), csp::UnsupportedSwitchType );
}

